// audiotap — stream what the Mac is playing to stdout, via a Core Audio process tap (macOS 14.2+).
//
//   audiotap --all                 everything that is playing (excluding nothing)
//   audiotap --bundle <substring>  only processes whose bundle id contains <substring>, e.g. "brave"
//   audiotap --list                print the processes Core Audio knows about, then exit
//
// Protocol: one line "rate <Hz>" on stderr, then raw mono Float32 samples on stdout until
// SIGTERM/SIGINT or the pipe closes. Warnings go to stderr as "warn: ...".
//
// Built by meeting_assistant/systemaudio.py with an embedded Info.plist: macOS attributes the
// "System Audio Recording" permission to that bundle id. Denied permission = silent zeros.

import CoreAudio
import Foundation

// MARK: - small Core Audio helpers

func fail(_ message: String) -> Never {
    fputs("error: \(message)\n", stderr)
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail("\(what) failed (OSStatus \(status))") }
}

func address(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
}

func readValue<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T) -> T? {
    var addr = address(selector)
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    let status = AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value)
    return status == noErr ? value : nil
}

func readString(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = address(selector)
    var value: CFString = "" as CFString
    var size = UInt32(MemoryLayout<CFString>.size)
    let status = AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &value)
    return status == noErr ? (value as String) : nil
}

func readObjectIDs(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector) -> [AudioObjectID] {
    var addr = address(selector)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(object, &addr, 0, nil, &size) == noErr, size > 0 else { return [] }
    var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
    guard AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &ids) == noErr else { return [] }
    return ids
}

struct AudioProcess {
    let object: AudioObjectID
    let pid: pid_t
    let bundleID: String
    let playing: Bool
}

func audioProcesses() -> [AudioProcess] {
    readObjectIDs(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyProcessObjectList).map { obj in
        AudioProcess(
            object: obj,
            pid: readValue(obj, kAudioProcessPropertyPID, pid_t(0)) ?? 0,
            bundleID: readString(obj, kAudioProcessPropertyBundleID) ?? "",
            playing: (readValue(obj, kAudioProcessPropertyIsRunningOutput, UInt32(0)) ?? 0) != 0
        )
    }
}

// MARK: - be our own "responsible process"
//
// TCC attributes a command-line tool's permission request to the app that launched it (a
// terminal, an editor) and judges that app's usage descriptions. Re-launching ourselves with
// the spawn attribute "disclaim responsibility" makes this binary — with its embedded
// Info.plist — the responsible process, so the prompt names the helper and the grant sticks
// to it no matter where it is run from.

var childPID: pid_t = 0

func respawnDisclaimed() {
    typealias SetDisclaim = @convention(c) (UnsafeMutablePointer<posix_spawnattr_t?>, Int32) -> Int32
    guard let sym = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "responsibility_spawnattrs_setdisclaim"),
          let exe = Bundle.main.executablePath else { return }
    let setDisclaim = unsafeBitCast(sym, to: SetDisclaim.self)
    var attr: posix_spawnattr_t? = nil
    guard posix_spawnattr_init(&attr) == 0, setDisclaim(&attr, 1) == 0 else { return }
    defer { posix_spawnattr_destroy(&attr) }

    let argv: [UnsafeMutablePointer<CChar>?] = CommandLine.arguments.map { strdup($0) } + [nil]
    var env = ProcessInfo.processInfo.environment
    env["AUDIOTAP_DISCLAIMED"] = "1"
    let envp: [UnsafeMutablePointer<CChar>?] = env.map { strdup("\($0.key)=\($0.value)") } + [nil]
    guard posix_spawn(&childPID, exe, nil, &attr, argv, envp) == 0 else { return }

    for sig in [SIGINT, SIGTERM] {
        signal(sig) { s in kill(childPID, s) }  // forward, so the real helper cleans up its tap
    }
    var status: Int32 = 0
    waitpid(childPID, &status, 0)
    exit((status & 0x7f) == 0 ? (status >> 8) & 0xff : 0)
}

if ProcessInfo.processInfo.environment["AUDIOTAP_DISCLAIMED"] == nil {
    respawnDisclaimed()  // returns only if the re-launch is impossible; then carry on in-process
}

// MARK: - permission ("System Audio Recording Only" under Privacy & Security)
//
// Core Audio never prompts by itself: without a grant, taps deliver silence. There is no
// public API to ask, so — like every open-source system-audio recorder — we use TCC's
// private request call. Status: 0 = granted, 1 = denied, anything else = not asked yet.

typealias TCCPreflight = @convention(c) (CFString, CFDictionary?) -> Int32
typealias TCCRequest = @convention(c) (CFString, CFDictionary?, @escaping @convention(block) (Bool) -> Void) -> Void

let audioCaptureService = "kTCCServiceAudioCapture" as CFString

func tccFunctions() -> (TCCPreflight, TCCRequest)? {
    guard let handle = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW),
          let pre = dlsym(handle, "TCCAccessPreflight"),
          let req = dlsym(handle, "TCCAccessRequest") else { return nil }
    return (unsafeBitCast(pre, to: TCCPreflight.self), unsafeBitCast(req, to: TCCRequest.self))
}

func permissionStatus() -> String {
    guard let (preflight, _) = tccFunctions() else { return "unknown" }
    switch preflight(audioCaptureService, nil) {
    case 0: return "granted"
    case 1: return "denied"
    default: return "not asked"
    }
}

let settingsHint = "allow it in System Settings › Privacy & Security › Screen & System Audio Recording (System Audio Recording Only), then retry"

func ensurePermission() {
    guard let (preflight, request) = tccFunctions() else {
        fputs("warn: cannot query the permission service; capture may be silent\n", stderr)
        return
    }
    if preflight(audioCaptureService, nil) == 0 { return }
    fputs("info: asking macOS for System Audio Recording permission — click Allow\n", stderr)
    let done = DispatchSemaphore(value: 0)
    var granted = false
    request(audioCaptureService, nil) { ok in
        granted = ok
        done.signal()
    }
    if done.wait(timeout: .now() + 180) == .timedOut {
        fail("no answer to the permission prompt; \(settingsHint)")
    }
    if !granted {
        fail("System Audio Recording permission denied; \(settingsHint)")
    }
}

// MARK: - output devices

func hasOutputStreams(_ device: AudioObjectID) -> Bool {
    var addr = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreams, mScope: kAudioObjectPropertyScopeOutput, mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    return AudioObjectGetPropertyDataSize(device, &addr, 0, nil, &size) == noErr && size > 0
}

/// The Mac's built-in output (speakers / headphone jack): a steady clock that is always running.
/// Bluetooth outputs only clock while something plays to them, and change rate when a call
/// opens their mic — a tap clocked by them delivers bursts or nothing (seen with Bluetooth earbuds).
func builtInOutput() -> AudioObjectID? {
    readObjectIDs(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDevices).first { device in
        hasOutputStreams(device)
            && (readValue(device, kAudioDevicePropertyTransportType, UInt32(0)) ?? 0) == kAudioDeviceTransportTypeBuiltIn
    }
}

// MARK: - arguments

var mode = "all"
var bundleFilter = ""
var clockChoice = "builtin"  // or "default": clock the tap by the current default output
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--all": mode = "all"
    case "--list": mode = "list"
    case "--check": mode = "check"
    case "--clock":
        guard !args.isEmpty else { fail("--clock needs builtin or default") }
        clockChoice = args.removeFirst()
    case "--bundle":
        guard !args.isEmpty else { fail("--bundle needs a value") }
        mode = "bundle"
        bundleFilter = args.removeFirst().lowercased()
    default: fail("unknown argument \(a)")
    }
}

if mode == "check" {
    print(permissionStatus())
    exit(0)
}
if mode == "list" {
    for p in audioProcesses() {
        print("pid \(p.pid)\t\(p.playing ? "playing" : "idle   ")\t\(p.bundleID)")
    }
    exit(0)
}
ensurePermission()

// MARK: - tap description

let description: CATapDescription
if mode == "bundle" {
    let matches = audioProcesses().filter { $0.bundleID.lowercased().contains(bundleFilter) }
    if matches.isEmpty {
        // Nothing with that bundle id has touched audio yet; fall back to everything rather than silence.
        description = CATapDescription(monoGlobalTapButExcludeProcesses: [])
        fputs("warn: no audio process matches \"\(bundleFilter)\" yet — capturing everything instead\n", stderr)
    } else {
        description = CATapDescription(monoMixdownOfProcesses: matches.map { $0.object })
        if #available(macOS 26.0, *) {
            // Follow these apps across restarts and newly spawned helper processes.
            description.bundleIDs = Array(Set(matches.map { $0.bundleID }))
            description.isProcessRestoreEnabled = true
        }
    }
} else {
    description = CATapDescription(monoGlobalTapButExcludeProcesses: [])
}
description.name = "Meeting Assistant tap"
description.uuid = UUID()
description.isPrivate = true
description.muteBehavior = .unmuted

var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(description, &tapID), "AudioHardwareCreateProcessTap")

guard let format = readValue(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription()) else {
    fail("could not read the tap's stream format")
}
let sampleRate = Int(format.mSampleRate)
let tapChannels = Int(format.mChannelsPerFrame)
let interleaved = format.mFormatFlags & kAudioFormatFlagIsNonInterleaved == 0
if format.mFormatFlags & kAudioFormatFlagIsFloat == 0 || format.mBitsPerChannel != 32 {
    fail("unexpected tap format (not Float32)")
}

// MARK: - aggregate device carrying the tap
//
// The tap captures the apps' audio wherever it is routed; the aggregate only needs a clock.
// The built-in output is used when there is one (steady, always running); otherwise the
// default output.

let systemObject = AudioObjectID(kAudioObjectSystemObject)
let defaultOutput = readValue(systemObject, kAudioHardwarePropertyDefaultOutputDevice, AudioObjectID(kAudioObjectUnknown))
let clockDevice = (clockChoice == "builtin" ? builtInOutput() : nil) ?? defaultOutput
guard let outputDevice = clockDevice, outputDevice != kAudioObjectUnknown,
      let outputUID = readString(outputDevice, kAudioDevicePropertyDeviceUID) else {
    fail("no output device to clock the tap")
}
let clockIsBuiltIn = outputDevice == builtInOutput()

let aggregateDescription: [String: Any] = [
    kAudioAggregateDeviceNameKey: "Meeting Assistant tap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
    kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true, kAudioSubTapUIDKey: description.uuid.uuidString]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggregateDescription as CFDictionary, &aggregateID), "AudioHardwareCreateAggregateDevice")

// MARK: - stream the tap's input to stdout

signal(SIGPIPE, SIG_IGN)  // a closed pipe (reader exited) ends us cleanly below
setvbuf(stdout, nil, _IOFBF, 1 << 16)

var procID: AudioDeviceIOProcID?
let ioQueue = DispatchQueue(label: "audiotap.io")
var mono = [Float32]()

check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, ioQueue) { _, inputData, _, _, _ in
    let buffers = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inputData))
    let count = buffers.count
    guard count > 0 else { return }
    // Sub-device inputs come first, tap streams last: take the tap's buffer(s) only.
    let tapBuffers = interleaved ? 1 : min(tapChannels, count)
    let first = count - tapBuffers
    guard let firstData = buffers[first].mData else { return }
    let frames = interleaved
        ? Int(buffers[first].mDataByteSize) / (MemoryLayout<Float32>.size * max(1, Int(buffers[first].mNumberChannels)))
        : Int(buffers[first].mDataByteSize) / MemoryLayout<Float32>.size
    guard frames > 0 else { return }
    if mono.count < frames { mono = [Float32](repeating: 0, count: frames) }
    if interleaved {
        let ch = max(1, Int(buffers[first].mNumberChannels))
        let samples = firstData.assumingMemoryBound(to: Float32.self)
        for f in 0..<frames {
            var acc: Float32 = 0
            for c in 0..<ch { acc += samples[f * ch + c] }
            mono[f] = acc / Float32(ch)
        }
    } else {
        for f in 0..<frames { mono[f] = 0 }
        for b in first..<count {
            guard let data = buffers[b].mData else { continue }
            let samples = data.assumingMemoryBound(to: Float32.self)
            for f in 0..<frames { mono[f] += samples[f] }
        }
        let n = Float32(count - first)
        for f in 0..<frames { mono[f] /= n }
    }
    let written = mono.withUnsafeBufferPointer { fwrite($0.baseAddress, MemoryLayout<Float32>.size, frames, stdout) }
    fflush(stdout)
    if written < frames { shutdown(0) }
}, "AudioDeviceCreateIOProcIDWithBlock")

func shutdown(_ code: Int32) -> Never {
    if let procID = procID {
        AudioDeviceStop(aggregateID, procID)
        AudioDeviceDestroyIOProcID(aggregateID, procID)
    }
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
    exit(code)
}

check(AudioDeviceStart(aggregateID, procID), "AudioDeviceStart")
fputs("rate \(sampleRate)\n", stderr)

// If the clock is the default output (no built-in output found), follow it: when the default
// changes or the clock device disappears, exit with code 3 so the app relaunches the tap.
let deviceQueue = DispatchQueue(label: "audiotap.devices")
var defaultOutputAddress = address(kAudioHardwarePropertyDefaultOutputDevice)
if !clockIsBuiltIn {
    AudioObjectAddPropertyListenerBlock(systemObject, &defaultOutputAddress, deviceQueue) { _, _ in
        fputs("warn: output device changed\n", stderr)
        shutdown(3)
    }
}
var aliveAddress = address(kAudioDevicePropertyDeviceIsAlive)
AudioObjectAddPropertyListenerBlock(outputDevice, &aliveAddress, deviceQueue) { _, _ in
    fputs("warn: output device went away\n", stderr)
    shutdown(3)
}

let signalQueue = DispatchQueue(label: "audiotap.signals")
for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: signalQueue)
    source.setEventHandler { shutdown(0) }
    source.resume()
    _ = Unmanaged.passRetained(source)  // keep alive for the life of the process
}
dispatchMain()

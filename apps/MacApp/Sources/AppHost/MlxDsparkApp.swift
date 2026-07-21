import AppCore
import SwiftUI

@main
struct MlxDsparkApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(model)
                .onAppear { delegate.model = model }
        }
        .windowResizability(.contentMinSize)
        .commands {
            CommandGroup(replacing: .newItem) { }      // single-window app
        }
    }
}

/// Owns app-lifetime concerns SwiftUI scenes can't express.
///
/// The important one is termination: the engine subprocess holds several GB of wired GPU
/// memory and a scheduler thread. Quitting without stopping it strands both until the OS
/// reaps the orphan, so shutdown is given a bounded window to run synchronously.
final class AppDelegate: NSObject, NSApplicationDelegate {
    @MainActor var model: AppModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Without this the window can open behind whatever is in front — or on a different
        // Space entirely when the frontmost app is fullscreen, which looks exactly like the
        // app failing to launch.
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model = MainActor.assumeIsolated({ self.model }) else { return .terminateNow }
        Task { @MainActor in
            await model.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

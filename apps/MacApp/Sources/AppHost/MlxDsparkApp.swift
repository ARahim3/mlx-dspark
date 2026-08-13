import AppCore
import SwiftUI

@main
struct MlxDsparkApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup(id: "main") {
            RootView()
                .environmentObject(model)
                .tint(Theme.spark)
                .onAppear { delegate.model = model }
        }
        .windowResizability(.contentMinSize)
        // Charts need room; the OS default opens at the minimum size, which makes the Lab
        // look cramped on first launch.
        .defaultSize(width: 1120, height: 760)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { model.newChat() }
                    .keyboardShortcut("n")
            }
        }

        // Ambient telemetry in the system menu bar — MTPLX's most-liked, cheapest idea: the
        // live rate stays visible without the window open, and it doubles as the way back to
        // the window when it's been closed. `.menuBarExtraStyle(.window)` gives a small popover
        // rather than a plain menu, so it can show a real gauge.
        MenuBarExtra {
            MenuBarPanel()
                .environmentObject(model)
                .tint(Theme.spark)
        } label: {
            MenuBarLabel()
                .environmentObject(model)
        }
        .menuBarExtraStyle(.window)
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

    /// Closing the window must NOT quit: the flagship use case is serving a coding agent in
    /// the background, and tearing the engine down under a live Claude Code session because
    /// the user tidied their windows would be the worst possible surprise. The app stays in
    /// the menu bar (which is also the way back to the window); quitting is ⌘Q or the menu
    /// bar's Quit, both of which stop the engine cleanly.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    /// Clicking the Dock icon with no window open should bring the window back.
    func applicationShouldHandleReopen(_ sender: NSApplication,
                                       hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            for window in NSApp.windows where window.canBecomeMain {
                window.makeKeyAndOrderFront(nil)
            }
        }
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let model = MainActor.assumeIsolated({ self.model }) else { return .terminateNow }
        Task { @MainActor in
            await model.shutdown()
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
}

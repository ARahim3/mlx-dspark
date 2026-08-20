import Darwin
import Foundation
import Security

/// Small, model-free helpers shared by the server supervisor and Settings UI.
public enum ServerNetworking {
    /// Every active, non-loopback IPv4 address that a client on the LAN can use.
    public static func lanIPv4Addresses() -> [String] {
        var interfaces: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&interfaces) == 0, let first = interfaces else { return [] }
        defer { freeifaddrs(interfaces) }

        var addresses = Set<String>()
        var current: UnsafeMutablePointer<ifaddrs>? = first
        while let interface = current {
            defer { current = interface.pointee.ifa_next }
            guard let address = interface.pointee.ifa_addr,
                  address.pointee.sa_family == UInt8(AF_INET) else { continue }

            let flags = Int32(interface.pointee.ifa_flags)
            guard flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }

            var addr = address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) {
                $0.pointee.sin_addr
            }
            var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
            guard inet_ntop(AF_INET, &addr, &buffer, socklen_t(INET_ADDRSTRLEN)) != nil else {
                continue
            }
            let value = String(cString: buffer)
            if value != "0.0.0.0" { addresses.insert(value) }
        }
        return addresses.sorted()
    }

    /// A 256-bit URL-safe token suitable for the server's bearer/x-api-key check.
    public static func generateAPIKey() -> String {
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = bytes.withUnsafeMutableBytes { raw in
            SecRandomCopyBytes(kSecRandomDefault, raw.count, raw.baseAddress!)
        }
        precondition(status == errSecSuccess, "Secure random generation failed")
        return Data(bytes).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}

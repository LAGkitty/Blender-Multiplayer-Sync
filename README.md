# Blender Multiplayer Sync

A Blender addon for real-time multiplayer collaboration and scene synchronization.

## Features

- Real-time object synchronization
- Mesh edit syncing
- Material syncing
- Camera and cursor sync
- LAN multiplayer support
- Internet support with port forwarding or VPN tools
- Built-in Blender UI panel

## Installation

1. Download `blender_multiplayer.py`
2. Open Blender
3. Go to `Edit > Preferences > Add-ons`
4. Click `Install...`
5. Select `blender_multiplayer.py`
6. Enable the addon

## Usage

1. Open the `Multiplayer` tab in the 3D View side panel
2. Enter your username
3. Host or join a session
4. Share the generated room code
5. Start collaborating live

## Networking

Default TCP port: `19283`

LAN works automatically.

For internet connections:
- Port forward TCP 19283
- Or use Tailscale / ZeroTier

## Blender Support

Designed for Blender 3.x

## License

MIT License

# Application

This layer is MUFASA's offline product surface. The Tauri desktop app is the primary host and controller: it starts the laptop-local service, model and retrieval package. A paired phone gets the same complete Ask, Compare and Coverage experience through a lightweight mobile web app over local Wi-Fi.

The phone performs no inference or retrieval and stores no model, database, graph or corpus. It sends requests to the laptop and receives progress, validated answers and selected evidence excerpts. No cloud or internet connection is required.

![MUFASA application architecture](./images/application-architecture.svg)

See [application-architecture.md](./application-architecture.md) for the desktop, mobile and laptop-local service design.

Packaged binaries and generated application bundles are release artifacts and are not stored in Git.

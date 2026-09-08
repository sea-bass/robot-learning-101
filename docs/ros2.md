# ROS 2 turtlesim driven by an LLM

Code: [`ros2_turtlesim/`](../ros2_turtlesim) (its README maps out the files).

This follows the [turtlesim tutorial](https://github.com/robotmcp/ros-mcp-server/tree/main/examples/1_turtlesim) of [ros-mcp-server](https://github.com/robotmcp/ros-mcp-server), an MCP server that lets a language model talk to any ROS system through rosbridge.
Nothing on the ROS side changes; you add one rosbridge node and the model gets tools to list topics, subscribe, publish and call services.
Turtlesim is the "hello world" robot for this: small enough that every topic and service fits on one screen.

The difference from the upstream tutorial is the install.
[RoboStack](https://robostack.github.io) publishes ROS as conda packages, so pixi pulls ROS 2 Jazzy, turtlesim and rosbridge into `.pixi/` and activates them for every `pixi run`.
There is no `apt install`, no `/opt/ros`, and no `source setup.bash`.
The MCP server itself is the `ros-mcp` package from PyPI, in the same environment.

## Setup

Everything runs through the `ros` pixi environment defined in the repository's [`pixi.toml`](../pixi.toml).
The first `pixi run` of a `ros-*` task builds it (about 3 GB on disk).
For any other ROS command, name the environment: `pixi run -e ros ros2 topic list`.
The turtlesim window needs a display; for headless runs set `QT_QPA_PLATFORM=offscreen`.

## 1. Launch turtlesim

```bash
pixi run ros-sim
```

A window appears with a turtle in the center of a blue background.

## 2. Explore topics and services

In a second terminal, look at what the running system exposes.
This is worth slowing down on: the model will discover exactly the same things through its tools in step 4.

```bash
pixi run -e ros ros2 topic list
pixi run -e ros ros2 topic echo /turtle1/pose --once
pixi run -e ros ros2 service list
pixi run -e ros ros2 service type /turtle1/set_pen
pixi run -e ros ros2 interface show geometry_msgs/msg/Twist
```

Topics are streams: turtlesim publishes the pose on `/turtle1/pose` and listens for velocity commands on `/turtle1/cmd_vel`.
Services are calls with a reply: `/spawn` adds a turtle, `/turtle1/set_pen` changes what it draws.

## 3. Start rosbridge and register the MCP server

rosbridge exposes the ROS graph over a websocket on port 9090.
Start it next to turtlesim, or start both at once with the launch file:

```bash
pixi run ros-bridge      # alongside a running ros-sim
pixi run ros-launch      # or: turtlesim + rosbridge together
```

Register the MCP server with Claude Code once, from anywhere, pointing at this repository's manifest:

```bash
claude mcp add ros-mcp -- pixi run --manifest-path /abs/path/to/robot-learning-101/pixi.toml ros-mcp-server
```

Launching through `pixi run` matters.
MCP clients start servers with a minimal environment, and only pixi's activation puts the right Python on the path.
Other clients (Claude Desktop, Cursor, Gemini CLI) take the same command in their MCP config; see the upstream [installation guide](https://github.com/robotmcp/ros-mcp-server/blob/main/docs/install/installation.md).

## 4. Drive the turtle in natural language

Open a Claude session and start with:

```
Connect to the robot at 127.0.0.1:9090 and tell me what it is.
```

The server exposes 31 tools.
Watch which ones the model calls: `detect_ros_version`, `get_topics`, `get_topic_type`, `get_message_details`, then `publish_for_durations` or `call_service`.
It has to read the message definitions before it can publish, the same way you did in step 2.

Then try the prompts from the upstream tutorial, roughly in order of difficulty:

```
What is the turtle's current position?
Move the turtle forward, then turn left.
Change the pen color to red and draw a square.
Spawn a second turtle at (2, 2) and make it follow a circle.
Move turtle1 to position (5, 5).
```

The last one is the interesting failure case.
There is no "go to" service, so the model has to close the loop itself: read the pose, publish a velocity, read again.
How well it manages that with a slow tool loop says a lot about where language models sit in a robot stack.

## Notes

- Verified headless with `QT_QPA_PLATFORM=offscreen` through the full chain: an MCP client connected via rosbridge, read the pose, published a velocity that moved the turtle, and spawned a turtle through `/spawn`.
- `fastmcp` is pinned below 3 because `ros-mcp` 3.1 imports 2.x internals; the resolver otherwise picks 4.x and the server fails at import.
- The manifest sets `PYTHONNOUSERSITE=1` so packages in `~/.local/lib/python3.12/site-packages` cannot shadow the environment; an old user-site `redis` broke the MCP server that way.

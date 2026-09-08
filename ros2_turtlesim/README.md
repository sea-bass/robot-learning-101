# ROS 2 turtlesim driven by an LLM

Guide: [`docs/ros2.md`](../docs/ros2.md).

```
launch/turtlesim_rosbridge.launch.py   turtlesim_node + rosbridge websocket (port:=9090), the
                                       two ROS processes the ros-mcp server needs
```

turtlesim, rosbridge and the `ros-mcp` MCP server come from the `ros` pixi environment
defined in the repository's `pixi.toml`; run the `ros-*` tasks.

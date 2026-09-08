"""Start turtlesim and rosbridge together, ready for the ros-mcp server.

Mirrors examples/1_turtlesim/ros_mcp_turtlesim.launch.py from
https://github.com/robotmcp/ros-mcp-server.

    pixi run ros-launch
    pixi run ros-launch port:=9091
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="9090",
                              description="rosbridge websocket port"),
        Node(package="turtlesim", executable="turtlesim_node", name="turtlesim",
             output="screen"),
        Node(package="rosbridge_server", executable="rosbridge_websocket",
             name="rosbridge_websocket", output="screen",
             parameters=[{
                 "port": LaunchConfiguration("port"),
                 # settings ros-mcp's own example uses
                 "use_compression": False,
                 "max_message_size": 10_000_000,
                 "send_action_goals_in_new_thread": True,
                 "call_services_in_new_thread": True,
             }]),
    ])

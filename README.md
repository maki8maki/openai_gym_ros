# openai_gym_ros

This package includes the openai_ros package from The Construct.

http://wiki.ros.org/openai_ros

https://www.theconstructsim.com/using-openai-ros/

## Installation instructions

### 1. Clone the repo and change some settings


```cd openai_gym_ros_ws/src```

```git clone https://github.com/maki8maki/openai_gym_ros.git```

``` rosdep install --from-paths src --ignore-src -y ```

**Specify of ros_ws_path**

training/turtlebot3_training/config/turtlebot3_world_params.yaml:

```ros_ws_abspath: "/home/USER/openai_gym_ros_ws" ```


```
cd openai_gym_ros_ws
catkin build
source devel/setup.bash
```

### 2. Run examples 

**Turtlebot3**

<img src="./robots/img/rl_turtlebot3.png" width="200">

---

Environment: world

<img src="./robots/img/rl_world.png" width="200">

```roslaunch turtlebot3_training start_training_world.launch```

### 3. Create your own agent

In the openai_ros folder you have to change the following things:

1. openai_ros_common.py

If you don't want to have all robot models in your workspace you can provide a link here which will download all
your model when you lanuch the training. Afterwards you have to do a catkin_make and launch again.

```
if  package_name == "turtlebot_gazebo":

    url_git_1 = "https://bitbucket.org/theconstructcore/turtlebot.git"
    package_git = [url_git_1]
    package_to_branch_dict[url_git_1] = "kinetic-gazebo9"
```

2. robot_envs

- turtlebot3_env.py

Define path of your robot location:


```
ROSLauncher(rospackage_name="turtlebot3_gazebo",
            launch_file_name="put_robot_in_world.launch",
            ros_ws_abspath=ros_ws_abspath)
```

3. task_envs

- turtlebot3_world.py

Define path of your environment location:

```
ROSLauncher(rospackage_name="turtlebot3_gazebo",
            launch_file_name="start_world.launch",
            ros_ws_abspath=ros_ws_abspath)
```

- turtlebot3_world.yaml

- task_envs_list.py

In this list you have to set your envrionment variable:


```
if task_env == 'TurtleBot3World-v0':
    register(
        id=task_env,
        entry_point='openai_ros.task_envs.turtlebot3.turtlebot3_world:TurtleBot3WorldEnv',
        max_episode_steps=max_episode_steps,
    )

    # import our training environment
    from openai_ros.task_envs.turtlebot3 import turtlebot3_world
```

Afterwards it also has to be set in the yaml file which is located in the trainings folder:

turtlebot3_world.yaml:

```task_and_robot_environment_name: 'TurtleBot3World-v0' ```


Structure openai_ros subfolder:

```
└── openai_ros
    ├── controllers_connection.py
    ├── gazebo_connection.py
    ├── __init__.py
    ├── openai_ros_common.py
    ├── robot_envs
    │   ├── __init__.py
    │   └── turtlebot3_env.py
    ├── robot_gazebo_env.py
    └── task_envs
        ├── __init__.py
        ├── task_commons.py
        ├── task_envs_list.py
        └── turtlebot3
            ├── config
            │   └── turtlebot3_world.yaml
            ├── __init__.py
            └── turtlebot3_world.py
```

Structure of trainings folder:

```
└── training
    └── turtlebot3_training
        ├── CMakeLists.txt
        ├── config
        │   └── turtlebot3_world_.yaml
        ├── launch
        │   └── start_training_world.launch
        ├── package.xml
        └── scripts
            ├── qlearn.py
            └── start_training.py
```



### 4. Notes

For more examples and a detail description have a look at:

http://wiki.ros.org/openai_ros/TurtleBot2%20with%20openai_ros

The original openai_ros package - version2:

https://bitbucket.org/theconstructcore/openai_ros/src/version2/

```
git clone https://bitbucket.org/theconstructcore/openai_ros.git
cd openai_ros;git checkout version2
```

More examples:

https://bitbucket.org/theconstructcore/openai_examples_projects/src/master/

```
git clone https://bitbucket.org/theconstructcore/openai_examples_projects.git
```

turtlebot repo:

https://github.com/turtlebot/turtlebot

```
git clone https://github.com/turtlebot/turtlebot.git
```

turtlebot3 repo:

https://github.com/ROBOTIS-GIT/turtlebot3

```
git clone https://github.com/ROBOTIS-GIT/turtlebot3.git
```



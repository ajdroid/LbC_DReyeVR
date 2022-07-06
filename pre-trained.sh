export CARLA_ROOT=/scratch/abhijatb/Bosch22/carla.v10_1
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes_training/route_19.xml         # change to desired route
export TEAM_AGENT=image_agent.py                                    # no need to change
export TEAM_CONFIG=epoch_24.ckpt                                       # change path to checkpoint
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window

./run_agent.sh
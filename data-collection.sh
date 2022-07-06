export CARLA_ROOT=/scratch/abhijatb/Bosch22/carla.v10_1
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes_training/route_19.xml         # change to desired route
export TEAM_AGENT=auto_pilot.py                                     # no need to change
export TEAM_CONFIG=new_data_collect                                     # change path to save data
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window

./run_agent.sh
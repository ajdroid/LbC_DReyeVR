export CARLA_ROOT=/scratch/abhijatb/Bosch22/carla.harp_p13bd
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=leaderboard/data/route55.xml         # change to desired route
export TEAM_AGENT=replay_pilot.py                                     # no need to change
export TEAM_CONFIG=dreyevr_data_collect                                     # change path to save data
export HAS_DISPLAY=1                                                # set to 0 if you don't want a debug window

./run_dreyevr_agent.sh
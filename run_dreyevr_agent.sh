#!/bin/bash
export CARLA_ROOT=/scratch/abhijatb/Bosch22/carla.harp_p13bd  # change to where you installed CARLA
export PORT=2000  # change to port that CARLA is running on
export ROUTES=leaderboard/data/route11.xml   # change to desired route
# export ROUTES=leaderboard/data/routes_training/route_19.xml   # change to desired route
export TEAM_AGENT=image_agent.py  # no need to change
# export TEAM_CONFIG=/scratch/abhijatb/Bosch22/LbC_DReyeVR/checkpoints/lbc_dreyevr_firstpass_img/epoch=30.ckpt  # change path to checkpoint
export TEAM_CONFIG=/scratch/abhijatb/Bosch22/2020_CARLA_challenge/epoch_24.ckpt  # change path to checkpoint
# export TEAM_AGENT=map_agent.py  # no need to change
# export TEAM_CONFIG=/scratch/abhijatb/Bosch22/LbC_DReyeVR/checkpoints/lbc_dreyevr_firstpass_map/epoch=41.ckpt  # change path to checkpoint
export HAS_DISPLAY=1  # set to 0 if you don't want a debug window
SCENARIO_FILE=leaderboard/data/dreyevr/periph_study_scenarios.json

export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.13-py3.6-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/examples # for DReyeVR_utils


if [ -d "$TEAM_CONFIG" ]; then
    CHECKPOINT_ENDPOINT="$TEAM_CONFIG/$(basename $ROUTES .xml).txt"
else
    CHECKPOINT_ENDPOINT="$(dirname $TEAM_CONFIG)/$(basename $ROUTES .xml).txt"
fi

python3 leaderboard/leaderboard/leaderboard_evaluator_dreyevr.py \
--track=SENSORS \
--scenarios=${SCENARIO_FILE}  \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--routes=${ROUTES} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--port=${PORT}

echo "Done. See $CHECKPOINT_ENDPOINT for detailed results."

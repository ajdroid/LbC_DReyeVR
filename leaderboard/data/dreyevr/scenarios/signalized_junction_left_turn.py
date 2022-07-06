#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a left turn
"""

from six.moves.queue import Queue  # pylint: disable=relative-import,bad-option-value

import py_trees
import math
import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      ActorSource,
                                                                      ActorSink,
                                                                      WaypointFollower,
                                                                      TrafficLightStateSetterWithTime
                                                                      )
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (generate_target_waypoint,
                                           get_location_in_distance_from_wp,
                                           get_waypoint_in_distance)


class SignalizedJunctionLeftTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning left at signalized junction scenario,
    Traffic Scenario 08.

    This is a single ego vehicle scenario
    """

    timeout = 80  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self._target_vel = 6.9
        self._brake_value = 0.5
        self._ego_distance = 110
        self._traffic_light = None
        self._ego_traffic_light = None
        self._traffic_light_other = None
        self._other_actor_transform = None
        self._blackboard_queue_name = 'SignalizedJunctionLeftTurn/actor_flow_queue'
        self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
        self._initialized = True
        self._reference_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._reference_location)

        super(SignalizedJunctionLeftTurn, self).__init__("TurnLeftAtSignalizedJunction",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),  # - 500),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, self._other_actor_transform)
        first_vehicle.set_transform(first_vehicle_transform)
        self.other_actors.append(first_vehicle)


        second_vehicle_transform = first_vehicle_transform
        # get the waypoint at first vehicle location
        waypoint = self._map.get_waypoint(first_vehicle_transform.location)
        # move back some from first vehicle
        second_vehicle_waypoint, _ = get_waypoint_in_distance(waypoint, -8)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)
        # print(first_vehicle_transform, second_vehicle_transform)
        second_vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.tt', second_vehicle_transform)
        second_vehicle.set_transform(second_vehicle_transform)
        second_vehicle.set_simulate_physics(True)
        self.other_actors.append(second_vehicle)

        return

    def _create_behavior(self):
        """
        Hero vehicle is turning left in an urban area,
        at a signalized intersection, while other is actor coming straight.
        The hero actor may turn left either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        self._ego_traffic_light = \
            CarlaDataProvider.get_next_traffic_light_by_location(self._reference_location)
        self._traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        # if self._traffic_light is None or self._traffic_light_other is None:
        #     raise RuntimeError("No traffic light for the given location found")

        # self._traffic_light.set_state(carla.TrafficLightState.Green)
        # self._traffic_light.set_green_time(self.timeout)
        #
        # # other vehicle's traffic light
        # self._traffic_light_other.set_state(carla.TrafficLightState.Green)
        # self._traffic_light_other.set_green_time(self.timeout)

        sequence = py_trees.composites.Sequence("Signalized junction left turn: Sequence Behavior")

        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[1].get_location()), 0)
        # Generating waypoint list till next intersection
        plan_second = []
        wp_choice = target_waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            target_waypoint = wp_choice[0]
            plan_second.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        # # adding flow of actors -- this used to be the old behaviour
        # actor_source = ActorSource(
        #     ['vehicle.tesla.model3', 'vehicle.audi.tt'],
        #     self._other_actor_transform, 5, self._blackboard_queue_name, actor_limit=0)
        # # destroying flow of actors
        # actor_sink = ActorSink(plan[-1][0].transform.location, 10)

        # follow waypoints until next intersection
        move_first_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan,
                                      blackboard_queue_name=self._blackboard_queue_name, avoid_collision=True)

        move_second_actor = WaypointFollower(self.other_actors[1], self._target_vel, plan=plan_second,
                                            blackboard_queue_name=self._blackboard_queue_name, avoid_collision=True)

        # wait
        wait = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # Behavior tree
        parallel_behaviour_root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        parallel_behaviour_root.add_child(wait)
        # parallel_behaviour_root.add_child(actor_source)
        # parallel_behaviour_root.add_child(actor_sink)
        parallel_behaviour_root.add_child(move_second_actor)
        parallel_behaviour_root.add_child(move_first_actor)

        # sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))

        ego_in_range = InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicles[0],
            50,
            name="Waiting for ego to get close to scenario")

        ego_traffic_hack = TrafficLightStateSetterWithTime(
            actor=self._ego_traffic_light,
            state=carla.TrafficLightState.Green,
            duration=self.timeout,
            name="Change ego light to green S8")

        other_traffic_hack = TrafficLightStateSetterWithTime(
            actor=self._traffic_light_other,
            state=carla.TrafficLightState.Green,
            duration=self.timeout,
            name="Change opposite light to green S8")

        start_scenario = py_trees.composites.Sequence()
        start_scenario.add_child(ego_in_range)
        start_scenario.add_child(ego_traffic_hack)
        start_scenario.add_child(other_traffic_hack)

        sequence.add_child(start_scenario)
        sequence.add_child(parallel_behaviour_root)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[1]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()

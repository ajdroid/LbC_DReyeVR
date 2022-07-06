#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Collection of traffic scenarios where the ego vehicle (hero)
is making a right turn
"""

from __future__ import print_function

import sys

import py_trees

import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower,
                                                                      TrafficLightStateSetterWithTime)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (DriveDistance,
                                                                               InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (get_geometric_linear_intersection,
                                           get_crossing_point,
                                           generate_target_waypoint,
                                           get_waypoint_in_distance)


class SignalizedJunctionRightTurn(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario,
    Traffic Scenario 09.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()

        self._target_vel = 6.9
        self._brake_value = 0.5
        self._ego_distance = 40
        self._traffic_light = None
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        self._reference_location = config.trigger_points[0].location
        self._reference_waypoint = self._map.get_waypoint(self._reference_location)


        super(SignalizedJunctionRightTurn, self).__init__("HeroActorTurningRightAtSignalizedJunction",
                                                          ego_vehicles,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_red_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        first_vehicle.set_simulate_physics(True)
        self.other_actors.append(first_vehicle)

        # get the waypoint at first vehicle location
        waypoint = self._map.get_waypoint(first_vehicle_transform.location)
        # move back enough from first vehicle that some ppl may shoot the gap
        second_vehicle_waypoint, _ = get_waypoint_in_distance(waypoint, -12)
        second_vehicle_transform = carla.Transform(second_vehicle_waypoint.transform.location,
                                                   second_vehicle_waypoint.transform.rotation)
        # print(first_vehicle_transform, second_vehicle_transform)
        second_vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.tt', second_vehicle_transform)
        second_vehicle.set_transform(second_vehicle_transform)
        second_vehicle.set_simulate_physics(True)
        self.other_actors.append(second_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area,
        at a signalized intersection, while other actor coming straight
        from left.The hero actor may turn right either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        # todo: spawn vehicles far from the intersection on the right in blind spot, move them fast across
        # todo: make the vehicle disappear

        # get the relevant traffic lights
        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        self._ego_traffic_light = \
            CarlaDataProvider.get_next_traffic_light_by_location(self._reference_location)
        self._traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])
        crossing_point_dynamic = get_crossing_point(self.other_actors[0])
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToVehicle(self.other_actors[0], self.ego_vehicles[0], 5)

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

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

        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan)
        waypoint_follower_end = InTriggerDistanceToLocation(
            self.other_actors[0], plan[-1][0].transform.location, 10)

        move_actor_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_parallel.add_child(move_actor)
        move_actor_parallel.add_child(waypoint_follower_end)
        # stop other actor, todo: why does this need to stop?
        stop = StopVehicle(self.other_actors[0], self._brake_value)
        # end condition
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # traffic lights hack
        ego_in_range = InTriggerDistanceToVehicle(
            self.other_actors[0],
            self.ego_vehicles[0],
            50,
            name="Waiting for ego to get close to scenario S9")

        ego_traffic_hack = TrafficLightStateSetterWithTime(
            actor=self._ego_traffic_light,
            state=carla.TrafficLightState.Red,
            duration=self.timeout,
            name="Change ego light to red S9")

        other_traffic_hack = TrafficLightStateSetterWithTime(
            actor=self._traffic_light_other,
            state=carla.TrafficLightState.Green,
            duration=self.timeout,
            name="Change other_actor light to green S9")

        start_scenario = py_trees.composites.Sequence()
        start_scenario.add_child(ego_in_range)
        start_scenario.add_child(ego_traffic_hack)
        start_scenario.add_child(other_traffic_hack)



        # Behavior tree
        sequence = py_trees.composites.Sequence("Signalized junction right turn: Sequence Behavior")
        # sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(start_scenario)
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(move_actor_parallel)
        sequence.add_child(stop)
        sequence.add_child(end_condition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

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


class SignalizedJunctionRightTurn_original(BasicScenario):

    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario,
    Traffic Scenario 09.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._target_vel = 6.9
        self._brake_value = 0.5
        self._ego_distance = 40
        self._traffic_light = None
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout
        super(SignalizedJunctionRightTurn_original, self).__init__("HeroActorTurningRightAtSignalizedJunction",
                                                          ego_vehicles,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.ego_vehicles[0], False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_red_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area,
        at a signalized intersection, while other actor coming straight
        from left.The hero actor may turn right either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])
        crossing_point_dynamic = get_crossing_point(self.other_actors[0])
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToLocation(self.other_actors[0], crossing_point_dynamic, 5)

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

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

        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan)
        waypoint_follower_end = InTriggerDistanceToLocation(
            self.other_actors[0], plan[-1][0].transform.location, 10)

        move_actor_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_parallel.add_child(move_actor)
        move_actor_parallel.add_child(waypoint_follower_end)
        # stop other actor
        stop = StopVehicle(self.other_actors[0], self._brake_value)
        # end condition
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # Behavior tree
        sequence = py_trees.composites.Sequence()
        # sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(move_actor_parallel)
        sequence.add_child(stop)
        sequence.add_child(end_condition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

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

//
// Created by nobleo on 12-9-18.
//

#include <mbf_msgs/ExePathResult.h>
#include <path_tracking_pid/PidDebug.h>
#include <path_tracking_pid/PidFeedback.h>
#include <pluginlib/class_list_macros.h>
#include <tf2/utils.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <path_tracking_pid/path_tracking_pid_local_planner.hpp>
#include <string>
#include <vector>

#include "common.hpp"

// register planner as move_base and move_base plugins
PLUGINLIB_EXPORT_CLASS(
  path_tracking_pid::TrackingPidLocalPlanner, mbf_costmap_core::CostmapController)
PLUGINLIB_EXPORT_CLASS(
  path_tracking_pid::TrackingPidLocalPlanner, nav_core::BaseLocalPlanner)

namespace path_tracking_pid
{
namespace
{
constexpr double MAP_PARALLEL_THRESH = 0.2;
constexpr double DT_MAX = 1.5;

/**
 * Convert the plan from geometry message format to tf2 format.
 *
 * @param[in] plan Plan to convert.
 * @return Converted plan.
 */
std::vector<tf2::Transform> convert_plan(const std::vector<geometry_msgs::PoseStamped> & plan)
{
  auto result = std::vector<tf2::Transform>{};

  result.reserve(plan.size());
  std::transform(
    plan.cbegin(), plan.cend(), std::back_inserter(result),
    [](const geometry_msgs::PoseStamped & msg) { return tf2_convert<tf2::Transform>(msg.pose); });

  return result;
}

}  // namespace

void TrackingPidLocalPlanner::reconfigure_pid(path_tracking_pid::PidConfig & config)
{
  pid_controller_.configure(config);
  controller_debug_enabled_ = config.controller_debug_enabled;
}

void TrackingPidLocalPlanner::initialize(
  std::string name, tf2_ros::Buffer * tf, costmap_2d::Costmap2DROS * costmap)
{
  ros::NodeHandle nh("~/" + name);
  ros::NodeHandle gn;
  ROS_DEBUG("TrackingPidLocalPlanner::initialize(%s, ..., ...)", name.c_str());
  // setup dynamic reconfigure
  pid_server_ =
    std::make_unique<dynamic_reconfigure::Server<path_tracking_pid::PidConfig>>(config_mutex_, nh);
  pid_server_->setCallback(
    [this](auto & config, auto /*unused*/) { this->reconfigure_pid(config); });
  pid_controller_.setEnabled(false);

  bool holonomic_robot;
  nh.param<bool>("holonomic_robot", holonomic_robot, false);
  pid_controller_.setHolonomic(holonomic_robot);

  bool estimate_pose_angle;
  nh.param<bool>("estimate_pose_angle", estimate_pose_angle, false);
  pid_controller_.setEstimatePoseAngle(estimate_pose_angle);

  nh.param<std::string>("base_link_frame", base_link_frame_, "base_link");

  nh.param<bool>("use_tricycle_model", use_tricycle_model_, false);
  nh.param<std::string>("steered_wheel_frame", steered_wheel_frame_, "steer");

  visualization_ = std::make_unique<Visualization>(nh);
  debug_pub_ = nh.advertise<path_tracking_pid::PidDebug>("debug", 1);
  path_pub_ = nh.advertise<nav_msgs::Path>("visualization_path", 1, true);

  odom_helper_.setOdomTopic("odom");

  sub_vel_max_external_ =
    nh.subscribe("vel_max", 1, &TrackingPidLocalPlanner::velMaxExternalCallback, this);
  feedback_pub_ = nh.advertise<path_tracking_pid::PidFeedback>("feedback", 1);

  costmap_ros_ = costmap;
  costmap_ = costmap_ros_->getCostmap();

  map_frame_ = costmap->getGlobalFrameID();

  costmap_model_ = new base_local_planner::CostmapModel(*costmap_);

  tf_ = tf;

  initialized_ = true;
}

bool TrackingPidLocalPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped> & global_plan)
{
  if (!initialized_) {
    ROS_ERROR(
      "path_tracking_pid has not been initialized, please call initialize() before using this "
      "planner");
    return false;
  }

  global_plan_map_frame_ = global_plan;

  std::string path_frame = global_plan_map_frame_.at(0).header.frame_id;
  ROS_DEBUG("TrackingPidLocalPlanner::setPlan(%zu)", global_plan_map_frame_.size());
  ROS_DEBUG("Plan is defined in frame '%s'", path_frame.c_str());

  /* If frame of received plan is not equal to mbf-map_frame, translate first */
  if (map_frame_ != path_frame) {
    ROS_DEBUG(
      "Transforming plan since my global_frame = '%s' and my plan is in frame: '%s'",
      map_frame_.c_str(), path_frame.c_str());
    geometry_msgs::TransformStamped tf_transform;
    tf_transform = tf_->lookupTransform(map_frame_, path_frame, ros::Time(0));
    // Check alignment, when path-frame is severely mis-aligned show error
    double yaw;
    double pitch;
    double roll;
    tf2::getEulerYPR(tf_transform.transform.rotation, yaw, pitch, roll);
    if (std::fabs(pitch) > MAP_PARALLEL_THRESH || std::fabs(roll) > MAP_PARALLEL_THRESH) {
      ROS_ERROR(
        "Path is given in %s frame which is severly mis-aligned with our map-frame: %s",
        path_frame.c_str(), map_frame_.c_str());
    }
    for (auto & pose_stamped : global_plan_map_frame_) {
      tf2::doTransform(pose_stamped.pose, pose_stamped.pose, tf_transform);
      pose_stamped.header.frame_id = map_frame_;
      // 'Project' plan by removing z-component
      pose_stamped.pose.position.z = 0.0;
    }
  }

  if (controller_debug_enabled_) {
    nav_msgs::Path received_path;
    received_path.header = global_plan_map_frame_.at(0).header;
    received_path.poses = global_plan_map_frame_;
    path_pub_.publish(received_path);
  }

  try {
    ROS_DEBUG(
      "map_frame: %s, plan_frame: %s, base_link_frame: %s", map_frame_.c_str(), path_frame.c_str(),
      base_link_frame_.c_str());
    tfCurPoseStamped_ = tf_->lookupTransform(map_frame_, base_link_frame_, ros::Time(0));
  } catch (const tf2::TransformException & ex) {
    ROS_ERROR("Received an exception trying to transform: %s", ex.what());
    return false;
  }

  // Feasibility check, but only when not resuming with odom-vel
  nav_msgs::Odometry odometry;
  odom_helper_.getOdom(odometry);
  if (
    pid_controller_.getConfig().init_vel_method != Pid_Odom &&
    pid_controller_.getConfig().init_vel_max_diff >= 0.0 &&
    std::abs(odometry.twist.twist.linear.x - pid_controller_.getCurrentForwardVelocity()) >
      pid_controller_.getConfig().init_vel_max_diff) {
    ROS_ERROR(
      "Significant diff between odom (%f) and controller_state (%f) detected. Aborting!",
      odometry.twist.twist.linear.x, pid_controller_.getCurrentForwardVelocity());
    return false;
  }

  if (use_tricycle_model_) {
    try {
      ROS_DEBUG(
        "base_link_frame: %s, steered_wheel_frame: %s", base_link_frame_.c_str(),
        steered_wheel_frame_.c_str());
      tf_base_to_steered_wheel_stamped_ =
        tf_->lookupTransform(base_link_frame_, steered_wheel_frame_, ros::Time(0));
    } catch (const tf2::TransformException & ex) {
      ROS_ERROR("Received an exception trying to transform: %s", ex.what());
      ROS_ERROR(
        "Invalid transformation from base_link_frame to steered_wheel_frame. Tricycle model will "
        "be disabled");
      use_tricycle_model_ = false;
    }

    pid_controller_.setTricycleModel(
      use_tricycle_model_,
      tf2_convert<tf2::Transform>(tf_base_to_steered_wheel_stamped_.transform));

    // TODO(clopez): subscribe to steered wheel odom
    geometry_msgs::Twist steering_odom_twist;
    if (!pid_controller_.setPlan(
          tf2_convert<tf2::Transform>(tfCurPoseStamped_.transform), odometry.twist.twist,
          tf2_convert<tf2::Transform>(tf_base_to_steered_wheel_stamped_.transform),
          steering_odom_twist, convert_plan(global_plan_map_frame_))) {
      return false;
    }
  } else {
    if (!pid_controller_.setPlan(
          tf2_convert<tf2::Transform>(tfCurPoseStamped_.transform), odometry.twist.twist,
          convert_plan(global_plan_map_frame_))) {
      return false;
    }
  }

  pid_controller_.setEnabled(true);
  active_goal_ = true;
  prev_time_ = ros::Time(0);
  return true;
}

std::optional<geometry_msgs::Twist> TrackingPidLocalPlanner::computeVelocityCommands()
{
  ros::Time now = ros::Time::now();
  if (prev_time_.isZero()) {
    prev_time_ = now - prev_dt_;  // Initialisation round
  }
  ros::Duration dt = now - prev_time_;
  if (dt.isZero()) {
    ROS_ERROR_THROTTLE(
      5, "dt=0 detected, skipping loop(s). Possible overloaded cpu or simulating too fast");
    auto cmd_vel = geometry_msgs::Twist();
    cmd_vel.linear.x = pid_controller_.getCurrentForwardVelocity();
    cmd_vel.angular.z = pid_controller_.getCurrentYawVelocity();
    // At the first call of computeVelocityCommands() we can't calculate a cmd_vel. We can't return
    // false because of https://github.com/magazino/move_base_flex/issues/195 so the current
    // velocity is send instead.
    return cmd_vel;
  }
  if (dt < ros::Duration(0) || dt > ros::Duration(DT_MAX)) {
    ROS_ERROR("Invalid time increment: %f. Aborting", dt.toSec());
    return std::nullopt;
  }
  try {
    ROS_DEBUG("map_frame: %s, base_link_frame: %s", map_frame_.c_str(), base_link_frame_.c_str());
    tfCurPoseStamped_ = tf_->lookupTransform(map_frame_, base_link_frame_, ros::Time(0));
  } catch (const tf2::TransformException & ex) {
    ROS_ERROR("Received an exception trying to transform: %s", ex.what());
    active_goal_ = false;
    return std::nullopt;
  }

  // Handle obstacles
  if (pid_controller_.getConfig().anti_collision) {
    // Let's get the pose of the robot in the frame of the plan
    geometry_msgs::PoseStamped robot_pose;
    if (!costmap_ros_->getRobotPose(robot_pose)) {
      ROS_ERROR("Could not get robot pose");
      return std::nullopt;
    }

    const auto linear_vel = pid_controller_.getCurrentForwardVelocity();
    const auto angular_vel = pid_controller_.getCurrentYawVelocity();

    if (isCollisionImminent(robot_pose, linear_vel, angular_vel)) {
      pid_controller_.setVelMaxObstacle(0.0);
      ROS_DEBUG("TrackingPidLocalPlanner detected collision ahead!");
    } else if (pid_controller_.getConfig().obstacle_speed_reduction) {
      // double max_vel = pid_controller_.getConfig().max_x_vel;
      // double reduction_factor = static_cast<double>(cost) / costmap_2d::LETHAL_OBSTACLE;
      // double limit = max_vel * (1 - reduction_factor);
      // ROS_DEBUG("Cost: %d, factor: %f, limit: %f", cost, reduction_factor, limit);
      // pid_controller_.setVelMaxObstacle(limit);
      ROS_WARN_THROTTLE(
        1.0, "TrackingPidLocalPlanner obstacle_speed_reduction is not implemented yet!");
    } else {
      ROS_DEBUG("TrackingPidLocalPlanner No collision imminent!");
      pid_controller_.setVelMaxObstacle(INFINITY);  // set back to inf
    }
  } else {
    pid_controller_.setVelMaxObstacle(INFINITY);  // Can be disabled live, so set back to inf
  }

  nav_msgs::Odometry odometry;
  odom_helper_.getOdom(odometry);
  const auto update_result = pid_controller_.update_with_limits(
    tf2_convert<tf2::Transform>(tfCurPoseStamped_.transform), odometry.twist.twist, dt);

  path_tracking_pid::PidFeedback feedback_msg;
  feedback_msg.eda = ros::Duration(update_result.eda);
  feedback_msg.progress = update_result.progress;
  feedback_pub_.publish(feedback_msg);

  if (controller_debug_enabled_) {
    debug_pub_.publish(update_result.pid_debug);

    // publish rviz visualization
    std_msgs::Header header;
    header.stamp = now;
    header.frame_id = map_frame_;
    const auto tfCurPose = tf2_convert<tf2::Transform>(tfCurPoseStamped_.transform);
    visualization_->publishAxlePoint(header, tfCurPose);
    visualization_->publishControlPoint(header, pid_controller_.getCurrentWithCarrot());
    visualization_->publishGoalPoint(header, pid_controller_.getCurrentGoal());
    visualization_->publishPlanPoint(header, pid_controller_.getCurrentPosOnPlan());
  }

  prev_time_ = now;
  prev_dt_ =
    dt;  // Store last known valid dt for next cycles (https://github.com/magazino/move_base_flex/issues/195)

  // In order to stop when a cancel was requested, we just return an empty Twist message here
  return cancel_in_progress_ ? geometry_msgs::Twist() : update_result.velocity_command;
}

bool TrackingPidLocalPlanner::inCollision(
    const double & x,
    const double & y,
    const double & theta)
{
    unsigned int mx, my;

    if (!costmap_->worldToMap(x, y, mx, my)) {
        ROS_WARN_THROTTLE(1.0, "The dimensions of the costmap is too small to successfully check for "
        "collisions as far ahead as requested. Proceed at your own risk, slow the robot, or "
        "increase your costmap size.");
        return false;
    }

    // Positive if all the points lie outside the footprint, negative otherwise:
    // -1 if footprint covers at least a lethal obstacle cell, or
    // -2 if footprint covers at least a no-information cell, or
    // -3 if footprint is [partially] outside of the map
    const double footprint_cost = costmap_model_->footprintCost(
        x, y, theta, costmap_ros_->getRobotFootprint());

    // This condition considers three cases:
    //
    // 1. Lethal obstacles (footprint_cost == costmap_2d::LETHAL_OBSTACLE)
    // 2. Inflated obstacles (footprint_cost == costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
    // 3. Unknown obstacles (footprint_cost == costmap_2d::NO_INFORMATION)
    return footprint_cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
}

bool TrackingPidLocalPlanner::isCollisionImminent(
    const geometry_msgs::PoseStamped & robot_pose,
    const double & linear_vel, const double & angular_vel)
{
  geometry_msgs::Pose2D curr_pose;
  curr_pose.x = robot_pose.pose.position.x;
  curr_pose.y = robot_pose.pose.position.y;
  curr_pose.theta = tf2::getYaw(robot_pose.pose.orientation);

  // Check if the previous projection step is in collision.
  // This avoids issues with the footprint projection changing with the speed of the robot,
  // and the robot being able to drive closer to the obstacles.
  if (!projection_steps_.empty() && inCollision(
        projection_steps_.back().getOrigin().x(),
        projection_steps_.back().getOrigin().y(),
        tf2::getYaw(projection_steps_.back().getRotation()))) {
    ROS_DEBUG_THROTTLE(5.0, "Collision detected at last projection step");
    return true;
  } else {
    projection_steps_.clear();
  }

  const auto footprint = costmap_ros_->getRobotFootprint();

  // Calculate dynamic lookahead distances
  const auto velocity_vector = std::hypot(linear_vel, angular_vel);
  const auto projection_time = costmap_->getResolution() / velocity_vector;  // [s]

  // Only forward simulate within time requested
  bool in_collision = false;
  size_t i = 0;
  while (i * projection_time < pid_controller_.getConfig().collision_look_ahead_time) {
    i++;

    // Predict future pose (using second order midpoint method)
    const auto delta_theta = projection_time * angular_vel / 2;
    const double midpoint_yaw = curr_pose.theta + delta_theta;
    auto delta_pos = tf2::Matrix3x3(createQuaternionFromYaw(midpoint_yaw))
        * tf2::Vector3(linear_vel, 0, 0) * projection_time;

    // Apply velocity at curr_pose over distance
    curr_pose.x += delta_pos.x();
    curr_pose.y += delta_pos.y();
    curr_pose.theta += delta_theta;

    // Visualize
    // Project footprint forward
    tf2::Quaternion orientation;
    orientation.setRPY(0.0, 0.0, curr_pose.theta);
    projection_steps_.push_back(tf2::Transform(orientation, tf2::Vector3(curr_pose.x, curr_pose.y, 0.0)));

    // check for collision at the projected pose
    if (inCollision(curr_pose.x, curr_pose.y, curr_pose.theta)) {
      in_collision = true;
      break;
    }
  }

  // No collision detected
  // Visualise the projected footprints
  projectionFootprint(footprint, projection_steps_, visualization_, map_frame_);
  return in_collision;
}

boost::geometry::model::ring<geometry_msgs::Point> TrackingPidLocalPlanner::projectionFootprint(
  const std::vector<geometry_msgs::Point> & footprint,
  const std::vector<tf2::Transform> & projected_steps, std::unique_ptr<Visualization> & viz,
  const std::string viz_frame)
{
  std::vector<tf2::Vector3> projected_footprint_points;
  polygon_t previous_footprint_xy;
  polygon_t projected_polygon;
  for (const auto & projection_tf : projected_steps) {
    // Project footprint forward
    double x = projection_tf.getOrigin().x();
    double y = projection_tf.getOrigin().y();
    double yaw = tf2::getYaw(projection_tf.getRotation());

    // Project footprint forward
    std::vector<geometry_msgs::Point> footprint_proj;
    costmap_2d::transformFootprint(x, y, yaw, footprint, footprint_proj);

    // Append footprint to polygon
    polygon_t two_footprints = previous_footprint_xy;
    previous_footprint_xy.clear();
    for (const auto & point : footprint_proj) {
      boost::geometry::append(two_footprints, point);
      boost::geometry::append(previous_footprint_xy, point);
    }

    boost::geometry::correct(two_footprints);
    polygon_t two_footprint_hull;
    boost::geometry::convex_hull(two_footprints, two_footprint_hull);
    projected_polygon = union_(projected_polygon, two_footprint_hull);

    // Add footprint to marker
    geometry_msgs::Point previous_point = footprint_proj.back();
    for (const auto & point : footprint_proj) {
      projected_footprint_points.push_back(tf2_convert<tf2::Vector3>(previous_point));
      projected_footprint_points.push_back(tf2_convert<tf2::Vector3>(point));
      previous_point = point;
    }
  }

  std_msgs::Header header;
  header.stamp = ros::Time::now();
  header.frame_id = viz_frame;
  viz->publishCollisionFootprint(header, projected_footprint_points);

  return projected_polygon;
}

bool TrackingPidLocalPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
  std::string dummy_message;
  geometry_msgs::PoseStamped dummy_pose;
  geometry_msgs::TwistStamped dummy_velocity, cmd_vel_stamped;
  const uint32_t outcome = computeVelocityCommands(dummy_pose, dummy_velocity, cmd_vel_stamped, dummy_message);
  cmd_vel = cmd_vel_stamped.twist;
  return outcome == mbf_msgs::ExePathResult::SUCCESS;
}

uint32_t TrackingPidLocalPlanner::computeVelocityCommands(
  const geometry_msgs::PoseStamped & /* pose */, const geometry_msgs::TwistStamped & /* velocity */,
  geometry_msgs::TwistStamped & cmd_vel, std::string & message)
{
  if (!initialized_) {
    message = "path_tracking_pid has not been initialized, please call initialize() before using this "
              "planner";
    ROS_ERROR_STREAM(message);
    active_goal_ = false;
    return mbf_msgs::ExePathResult::NOT_INITIALIZED;
  }
  // TODO(Cesar): Use provided pose and odom
  const auto opt_cmd_vel = computeVelocityCommands();
  if (!opt_cmd_vel) {
    active_goal_ = false;
    return mbf_msgs::ExePathResult::FAILURE;
  }
  cmd_vel.twist = *opt_cmd_vel;
  cmd_vel.header.stamp = ros::Time::now();
  cmd_vel.header.frame_id = base_link_frame_;

  bool moving = std::abs(cmd_vel.twist.linear.x) > VELOCITY_EPS;
  if (cancel_in_progress_) {
    if (!moving) {
      message = "Cancel requested and we now (almost) reached velocity 0: " + std::to_string(cmd_vel.twist.linear.x);
      ROS_INFO_STREAM(message);
      cancel_in_progress_ = false;
      active_goal_ = false;
      return mbf_msgs::ExePathResult::CANCELED;
    }
    message = "Cancel in progress... remaining x_vel: " + std::to_string(cmd_vel.twist.linear.x);
    ROS_INFO_STREAM_THROTTLE(1.0, message);
    return to_underlying(ComputeVelocityCommandsResult::GRACEFULLY_CANCELLING);
  }

  if (!moving && pid_controller_.getVelMaxObstacle() < VELOCITY_EPS) {
    active_goal_ = false;
    return mbf_msgs::ExePathResult::BLOCKED_PATH;
  }

  if (isGoalReached()) {
    active_goal_ = false;
  }
  message = "Goal reached!";
  ROS_DEBUG_STREAM(message);
  return mbf_msgs::ExePathResult::SUCCESS;
}

bool TrackingPidLocalPlanner::isGoalReached()
{
  // Return reached boolean, but never succeed when we're preempting
  return pid_controller_.isEndReached() && !cancel_in_progress_;
}

bool TrackingPidLocalPlanner::isGoalReached(
  double /* dist_tolerance */, double /* angle_tolerance */)
{
  return isGoalReached();
}

bool TrackingPidLocalPlanner::cancel()
{
  // This function runs in a separate thread
  cancel_in_progress_ = true;
  ros::Rate r(10);
  ROS_INFO("Cancel requested, waiting in loop for cancel to finish");
  while (active_goal_) {
    r.sleep();
  }
  ROS_INFO("Finished waiting loop, done cancelling");
  return true;
}

void TrackingPidLocalPlanner::velMaxExternalCallback(const std_msgs::Float64 & msg)
{
  pid_controller_.setVelMaxExternal(msg.data);
}
}  // namespace path_tracking_pid

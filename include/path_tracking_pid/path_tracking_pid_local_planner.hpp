#pragma once

#include <base_local_planner/costmap_model.h>
#include <base_local_planner/odometry_helper_ros.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Pose2D.h>
#include <mbf_costmap_core/costmap_controller.h>
#include <nav_core/base_local_planner.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <path_tracking_pid/PidConfig.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <tf2_ros/buffer.h>

#include <atomic>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/register/point.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <path_tracking_pid/controller.hpp>
#include <path_tracking_pid/visualization.hpp>

BOOST_GEOMETRY_REGISTER_POINT_2D(geometry_msgs::Point, double, cs::cartesian, x, y)

namespace path_tracking_pid
{
class TrackingPidLocalPlanner : public mbf_costmap_core::CostmapController
                              , public nav_core::BaseLocalPlanner
                              , private boost::noncopyable
{
private:
  using polygon_t = boost::geometry::model::ring<geometry_msgs::Point>;

  static inline polygon_t union_(const polygon_t & polygon1, const polygon_t & polygon2)
  {
    std::vector<polygon_t> output_vec;
    boost::geometry::union_(polygon1, polygon2, output_vec);
    return output_vec.at(0);  // Only first vector element is filled
  }

public:
  /**
    * @brief Initialize local planner
    * @param name The name of the planner
    * @param tf a pointer to TransformListener in TF Buffer
    * @param costmap Costmap indicating free/occupied space
    */
  void initialize(
    std::string name, tf2_ros::Buffer * tf, costmap_2d::Costmap2DROS * costmap) override;

  /**
    * @brief Set the plan we should be following
    * @param global_plan Plan to follow as closely as we can
    * @return whether the plan was successfully updated or not
    */
  bool setPlan(const std::vector<geometry_msgs::PoseStamped> & global_plan) override;

  /**
   * @brief Calculates the velocity command based on the current robot pose given by pose. The velocity
   * and message are not set. See the interface in move base flex.
   * @param pose Current robot pose
   * @param velocity
   * @param cmd_vel Output the velocity command
   * @param message
   * @return a status code defined in the move base flex ExePath action.
   */
  uint32_t computeVelocityCommands(
    const geometry_msgs::PoseStamped & pose, const geometry_msgs::TwistStamped & velocity,
    geometry_msgs::TwistStamped & cmd_vel, std::string & message) override;

  /**
   * @brief  Given the current position, orientation, and velocity of the robot,
   * compute velocity commands to send to the base
   * @param cmd_vel Will be filled with the velocity command to be passed to the robot base
   * @return True if a valid trajectory was found, false otherwise
   */
  bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;

  /**
   * @brief Returns true, if the goal is reached. Currently does not respect the parameters given.
   * @param dist_tolerance Tolerance in distance to the goal
   * @param angle_tolerance Tolerance in the orientation to the goals orientation
   * @return true, if the goal is reached
   */
  bool isGoalReached(double dist_tolerance, double angle_tolerance) override;

  /**
   * @brief Cancels the planner.
   * @return True on cancel success.
   */
  bool cancel() override;

  /** Enumeration for custom SUCCESS feedback codes. See default ones:
   * https://github.com/magazino/move_base_flex/blob/master/mbf_msgs/action/ExePath.action
  */
  enum class ComputeVelocityCommandsResult { SUCCESS = 0, GRACEFULLY_CANCELLING = 1 };

private:
  /**
   * @brief Calculates the velocity command based on the current robot pose given by pose.
   * @return Velocity command on success, empty optional otherwise.
   */
  std::optional<geometry_msgs::Twist> computeVelocityCommands();

  /**
   * @brief Returns true, if the goal is reached.
   * @return true, if the goal is reached
   */
  bool isGoalReached() override;

  /**
   * Accept a new configuration for the PID controller
   * @param config the new configuration
   */
  void reconfigure_pid(path_tracking_pid::PidConfig & config);

  void pauseCallback(std_msgs::Bool pause);

  void curOdomCallback(const nav_msgs::Odometry & odom_msg);

  void velMaxExternalCallback(const std_msgs::Float64 & msg);

  /**
   * @brief Expand the footprint with the projected steps
   * @param footprint
   * @param projected_steps
   * @param viz Used for marker publishing
   * @param viz_frame Used for marker publishing
   * @return Projected footprint
   */
  static polygon_t projectionFootprint(
    const std::vector<geometry_msgs::Point> & footprint,
    const std::vector<tf2::Transform> & projected_steps, std::unique_ptr<Visualization> & viz,
    const std::string viz_frame);

  bool inCollision( const double & x, const double & y,
                    const double & theta);

  bool isCollisionImminent(
      const geometry_msgs::PoseStamped & robot_pose,
      const double & linear_vel, const double & angular_vel);

  tf2::Quaternion createQuaternionFromYaw(double yaw)
  {
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    return q;
  }

  nav_msgs::Odometry latest_odom_;
  ros::Time prev_time_;
  ros::Duration prev_dt_;
  path_tracking_pid::Controller pid_controller_;

  // Obstacle collision detection
  costmap_2d::Costmap2D * costmap_ = nullptr;
  costmap_2d::Costmap2DROS * costmap_ros_ = nullptr;
  // For retrieving robot footprint cost
  base_local_planner::CostmapModel* costmap_model_ = nullptr;

  // Cancel flags (multi threaded, so atomic bools)
  std::atomic<bool> active_goal_{false};
  std::atomic<bool> cancel_in_progress_{false};

  // dynamic reconfiguration
  boost::recursive_mutex config_mutex_;
  std::unique_ptr<dynamic_reconfigure::Server<path_tracking_pid::PidConfig>> pid_server_;

  tf2_ros::Buffer * tf_ = nullptr;
  geometry_msgs::TransformStamped tfCurPoseStamped_;

  ros::Publisher debug_pub_;  // Debugging of controller internal parameters

  // Rviz visualization
  std::unique_ptr<Visualization> visualization_;
  ros::Publisher path_pub_;

  ros::Publisher feedback_pub_;

  ros::Subscriber sub_vel_max_external_;

  std::string map_frame_;
  std::string base_link_frame_;
  bool initialized_ = false;

  // Used for tricycle model
  bool use_tricycle_model_ = false;
  std::string steered_wheel_frame_;
  geometry_msgs::TransformStamped tf_base_to_steered_wheel_stamped_;

  // Controller logic
  bool controller_debug_enabled_ = false;

  std::vector<geometry_msgs::PoseStamped> global_plan_map_frame_;

  base_local_planner::OdometryHelperRos odom_helper_;
};

}  // namespace path_tracking_pid

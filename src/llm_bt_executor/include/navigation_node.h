#include <behaviortree_cpp/action_node.h>

// This is an asynchronous operation
class NavigationAction : public BT::StatefulActionNode
{
public:
  // Any TreeNode with ports must have a constructor with this signature
  NavigationAction(const std::string& name, const BT::NodeConfig& config)
    : StatefulActionNode(name, config)
  {}

  // It is mandatory to define this static method.
  static BT::PortsList providedPorts()
  {
    return {BT::InputPort<std::string>("start_pos"),
            BT::InputPort<std::string>("end_pos")};
  }

  // this function is invoked once at the beginning.
  BT::NodeStatus onStart() override;

  // If onStart() returned RUNNING, we will keep calling
  // this method until it return something different from RUNNING
  BT::NodeStatus onRunning() override;

  // callback to execute if the action was aborted by another node
  void onHalted() override;

private:
  std::chrono::system_clock::time_point _completion_time;
  std::string start_pos_;
  std::string end_pos_;
};
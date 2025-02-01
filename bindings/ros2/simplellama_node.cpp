/*
 *  Copyright 2025 (C) Jeroen Veen <ducroq> & Victor Hogeweij <Hoog-V>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * This file is part of the SimpleLLama.Cpp library
 *
 * Author:         Jeroen Veen <ducroq>
 *                 Victor Hogeweij <Hoog-V>
 *
 */
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/set_bool.hpp"
#include <simplellama.hpp>

class SimpleLLamaNode : public rclcpp::Node
{
public:
    SimpleLLamaNode() : Node("simplellama_node")
    {
        declare_and_get_parameters();

        setup_communication();
        sl_ = std::make_unique<SimpleLLama>(model_params_);
        sl_->init();
        llm_engine_enabled = true;
    }

~SimpleLLamaNode()
{
    
}
private:
    void declare_and_get_parameters()
    {
        declare_parameter("text_input_topic", "/llm_input");
        declare_parameter("text_output_topic", "/llm_output");
        declare_parameter("toggle_topic", "/llm_toggle");
        declare_parameter("gguf_model_path", "phi-2.Q5_0.gguf");

        text_in_topic_ = get_parameter("text_input_topic").as_string();
        text_out_topic_ = get_parameter("text_output_topic").as_string();
        toggle_topic_ = get_parameter("toggle_topic").as_string();
        model_params_.model_llama = get_parameter("gguf_model_path").as_string();
    }

    void setup_communication()
    {
        text_in_sub_ = create_subscription<std_msgs::msg::String>(
            text_in_topic_, rclcpp::SensorDataQoS(),
            std::bind(&SimpleLLamaNode::text_in_cb, this, std::placeholders::_1));

        text_out_pub_ = create_publisher<std_msgs::msg::String>(text_out_topic_, 10);
        toggle_in_sub_ = create_service<std_srvs::srv::SetBool>(toggle_topic_,
                                                                std::bind(&SimpleLLamaNode::toggle_in_cb, this, std::placeholders::_1, std::placeholders::_2));
    }

    void text_in_cb(const std_msgs::msg::String::SharedPtr msg)
    {
        if (!llm_engine_enabled)
            return;

        std_msgs::msg::String llm_text = std_msgs::msg::String();
        llm_text.data = sl_->do_inference(msg->data);
        text_out_pub_->publish(llm_text);
    }

    void toggle_in_cb(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                      std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        llm_engine_enabled = request->data;
        response->success = true;
        response->message = llm_engine_enabled ? "llm engine enabled" : "llm engine disabled";
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr text_in_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_out_pub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr toggle_in_sub_;
    simplellama_model_params model_params_;
    std::unique_ptr<SimpleLLama> sl_;
    std::string text_in_topic_;
    std::string text_out_topic_;
    std::string toggle_topic_;
    bool llm_engine_enabled;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SimpleLLamaNode>());
    rclcpp::shutdown();
    return 0;
}
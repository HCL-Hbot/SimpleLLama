#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <vector>
#include <memory>

class SimpleLLamaDemo : public rclcpp::Node
{
public:
    SimpleLLamaDemo() : Node("simplellama_demo"), question_index_(0), waiting_for_response_(false)
    {
        publisher_ = this->create_publisher<std_msgs::msg::String>("/llm_input", 10);
        subscription_ = this->create_subscription<std_msgs::msg::String>(
            "/llm_output", 10,
            std::bind(&SimpleLLamaDemo::response_callback, this, std::placeholders::_1));
        
        questions_ = {"What came first, the egg or the chicken?",
                      "What is the capital city of Germany?",
                      "What is the capital city of the United States?",
                      "What is 10 * 100 = ?",
                      "What is the speed of light in vacuum?"};

        // Ask the first question after the node is initialized.
        this->declare_parameter("start_questions", true);
        this->get_parameter("start_questions", start_questions_);
        
        if (start_questions_) {
            ask_question();
        }
    }

private:
    void ask_question()
    {
        if (question_index_ < questions_.size() && !waiting_for_response_)
        {
            auto message = std_msgs::msg::String();
            message.data = questions_[question_index_];
            RCLCPP_INFO(this->get_logger(), "Asking: %s", message.data.c_str());
            publisher_->publish(message);
            waiting_for_response_ = true;  // Prevent sending the next question before response arrives
        }
        else if (question_index_ >= questions_.size())
        {
            RCLCPP_INFO(this->get_logger(), "All questions asked.");
            rclcpp::shutdown();
        }
    }

    void response_callback(const std_msgs::msg::String::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Received response: %s", msg->data.c_str());
        question_index_++;
        waiting_for_response_ = false;  // Reset flag, allowing next question to be asked
        ask_question();  // Now, send the next question
    }

    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
    std::vector<std::string> questions_;
    size_t question_index_;
    bool waiting_for_response_;
    bool start_questions_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SimpleLLamaDemo>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

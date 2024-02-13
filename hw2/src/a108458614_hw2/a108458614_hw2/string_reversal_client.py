import sys
import rclpy
from rclpy.node import Node
from a108458614_service.srv import ProcessString
import matplotlib.pyplot as plt
import time

class StringReversalClient(Node):
    def __init__(self):
        super().__init__('string_reversal_client')
        self.client = self.create_client(ProcessString, 'reverse_string')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = ProcessString.Request()

    def send_request(self, input_string):
        self.req.input = input_string
        self.future = self.client.call_async(self.req)

def main(args=None):
    rclpy.init(args=args)
    string_reversal_client = StringReversalClient()

    adjusted_rtts = []

    for i in range(400):
        input_string = f'Message {i}'
        start_time = time.time()

        string_reversal_client.send_request(input_string)
        rclpy.spin_until_future_complete(string_reversal_client, string_reversal_client.future)

        end_time = time.time()
        rtt = end_time - start_time

        try:
            response = string_reversal_client.future.result()
            print('response: ', response.output_str, response.output_duration)
            service_time = response.output_duration
            adjusted_rtt = service_time
            adjusted_rtts.append(adjusted_rtt)
        except Exception as e:
            string_reversal_client.get_logger().info(f'Service call failed: {e}')

    plt.hist(adjusted_rtts, bins=30, edgecolor='black')
    plt.title('Histogram of Service Node Wait Time (400 calls)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
import re


def read_accuracy_data(filename):
    approx_bits = []
    accuracy = []

    with open(filename, "r") as file:
        for line in file:

            accuracy_match = re.search(r"Test Accuracy: (\d+\.\d+)%", line)
            bits_match = re.search(r"approx_bits:(\d+)", line)

            if accuracy_match and bits_match:

                approx_bits.append(int(bits_match.group(1)))
                accuracy.append(float(accuracy_match.group(1)))

    return approx_bits, accuracy


if __name__ == "__main__":

    filename = input("The log file: ")
    try:
        x, y = read_accuracy_data(filename)

        plt.figure(figsize=(12, 6))
        plt.plot(x, y, marker="o", linestyle="-", color="b", label="Test Accuracy")

        plt.title(f"TypeC Approx adder on addernet with int32 quantization", fontsize=16)
        plt.xlabel("Approx Bits", fontsize=12)
        plt.ylabel("Test Accuracy (%)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)

        plt.xticks(range(min(x), max(x) + 1, 1))

        plt.legend()

        plt.tight_layout()

        plt.show()
        plt.savefig('accuracy_vs_approx_bits.png', dpi=300, bbox_inches='tight')

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

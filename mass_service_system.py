import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Define parameters
SOCIAL_MEDIA_ADVERTISING_ARRIVAL_RATE = 50
# Processing rate for Social Media Advertising System
PROCESSING_RATE_SOCIAL_MEDIA_ADVERTISING = 80
# Processing rate for Offer Conversion System
PROCESSING_RATE_OFFER_CONVERSION_SYSTEM = 500
NUM_CLIENTS = 1000
# Probability of user transitioning to Offer Conversion System
TRANSITION_PROBABILITY = 0.1

# Data for analytical calculations
load_factors_social_media = np.linspace(1 / 9, 8 / 9, 8)
load_factors_offer_conversion = np.linspace(1 / 8, 7 / 8, 7)

# For collecting statistics
waiting_times_social_media = []
waiting_times_offer_conversion = []
processed_clients = set()  # Unique clients who transitioned to Offer Conversion System


def social_media_advertising(env, social_media_queue, offer_conversion_queue, client_id):
    """Social Media Advertising System process"""
    arrival_time = env.now
    yield env.timeout(random.expovariate(1 / SOCIAL_MEDIA_ADVERTISING_ARRIVAL_RATE))

    with social_media_queue.request() as request:
        yield request  # Waiting for resource (queue)

        queue_wait_time = env.now - arrival_time
        waiting_times_social_media.append(queue_wait_time)

        yield env.timeout(random.expovariate(1 / PROCESSING_RATE_SOCIAL_MEDIA_ADVERTISING))

    # Transition to Offer Conversion System after processing in Social Media Advertising
    if random.random() < TRANSITION_PROBABILITY:
        if client_id not in processed_clients:  # Check if the client has already been processed
            processed_clients.add(client_id)  # Add unique ID
            # Only after processing Social Media Advertising
            # Transition to Offer Conversion System
            env.process(offer_conversion_system(
                env, offer_conversion_queue, client_id))
            print(
                f"Client {client_id} transitioned to Offer Conversion System")


def offer_conversion_system(env, offer_conversion_queue, client_id):
    """Offer Conversion System process"""
    arrival_time = env.now

    with offer_conversion_queue.request() as request:
        yield request  # Waiting for resource (queue)

        # Processing time in Offer Conversion System
        queue_wait_time = env.now - arrival_time
        waiting_times_offer_conversion.append(queue_wait_time)

        # Use new processing speed for Offer Conversion System
        yield env.timeout(random.expovariate(1 / PROCESSING_RATE_OFFER_CONVERSION_SYSTEM))


def setup(env):
    """Simulation setup"""
    social_media_queue = simpy.Resource(env, capacity=1)
    offer_conversion_queue = simpy.Resource(env, capacity=1)

    # Create process for each client
    for client_id in range(NUM_CLIENTS):
        env.process(social_media_advertising(
            env, social_media_queue, offer_conversion_queue, client_id))
        yield env.timeout(random.expovariate(1 / SOCIAL_MEDIA_ADVERTISING_ARRIVAL_RATE))


def calculate_analytical_values(load_factors, system_name):
    """Calculate analytical values n and l."""
    analytical_table = [
        ["Load Factor (ρ)", "Queue Length (Analytical, l)", "Number of Requests in System (Analytical, n)"]]
    analytical_lengths = []
    analytical_requests = []

    for p in load_factors:
        if p < 1:
            n = p / (1 - p)
            l = (p ** 2) / (1 - p)
            analytical_table.append([f"{p:.2f}", f"{l:.2f}", f"{n:.2f}"])
            analytical_lengths.append(l)
            analytical_requests.append(n)

    print(f"\nAnalytical results for {system_name}:")
    print(tabulate(analytical_table[1:],
          headers=analytical_table[0], tablefmt="grid"))

    return analytical_lengths, analytical_requests


def plot_analytical_graphs(load_factors, analytical_lengths, analytical_requests, system_name, subplot_position):
    """Plot analytical queue length (l) and number of requests (n)."""
    plt.subplot(3, 2, subplot_position)

    plt.plot(load_factors[:len(analytical_lengths)], analytical_lengths,
             label=f"Analytical Queue Length (l) {system_name}", marker='o')
    plt.plot(load_factors[:len(analytical_requests)], analytical_requests,
             label=f"Analytical Number of Requests (n) {system_name}", marker='x')

    plt.title(f"{system_name}")
    plt.legend()


def plot_regression_graph(load_factors, analytical_lengths, system_name, subplot_position):
    """Plot regression graph for analytical queue length."""
    plt.subplot(3, 2, subplot_position)
    plt.plot(load_factors[:len(analytical_lengths)], analytical_lengths,
             label=f"Analytical Queue Length {system_name}", marker='o')

    x = np.array(load_factors[:len(analytical_lengths)])
    y = np.array(analytical_lengths)
    xi = np.linspace(min(x), max(x), 100)

    coeff1 = np.polyfit(x, y, 1)
    y1 = np.polyval(coeff1, xi)
    plt.plot(
        xi, y1, label=f"1st Degree Regression {system_name}", linestyle="--")

    coeff2 = np.polyfit(x, y, 2)
    y2 = np.polyval(coeff2, xi)
    plt.plot(
        xi, y2, label=f"2nd Degree Regression {system_name}", linestyle=":")

    plt.title(f"{system_name}")
    plt.legend()


def plot_histogram(waiting_times, system_name, subplot_position):
    """Plot histogram of waiting times in queue."""
    plt.subplot(3, 2, subplot_position)
    plt.hist(waiting_times, bins=20, alpha=0.7,
             color='blue', edgecolor='black')
    plt.title(f"Queue Histogram {system_name}")
    plt.xlabel("Waiting Time")
    plt.ylabel("Frequency")


# Simulation
env = simpy.Environment()
env.process(setup(env))
env.run()

# Check the number of clients that transitioned to Offer Conversion System
print(
    f"Number of clients who transitioned to Offer Conversion System: {len(processed_clients)}")

# Analytical calculations
analytical_lengths_social_media, analytical_requests_social_media = calculate_analytical_values(
    load_factors_social_media, "Social Media Advertising System")
analytical_lengths_offer_conversion, analytical_requests_offer_conversion = calculate_analytical_values(
    load_factors_offer_conversion, "Offer Conversion System")

# Plotting graphs
plt.figure(figsize=(14, 10))

# Analytical data graphs
plot_analytical_graphs(load_factors_social_media, analytical_lengths_social_media,
                       analytical_requests_social_media, "Social Media Advertising System", 1)
plot_analytical_graphs(load_factors_offer_conversion, analytical_lengths_offer_conversion,
                       analytical_requests_offer_conversion, "Offer Conversion System", 2)

# Regression graphs
plot_regression_graph(load_factors_social_media, analytical_lengths_social_media,
                      "Social Media Advertising System", 3)
plot_regression_graph(load_factors_offer_conversion, analytical_lengths_offer_conversion,
                      "Offer Conversion System", 4)

# Histograms
plot_histogram(waiting_times_social_media,
               "Social Media Advertising System", 5)
plot_histogram(waiting_times_offer_conversion, "Offer Conversion System", 6)

plt.tight_layout()
plt.show()

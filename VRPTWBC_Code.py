import pandas as pd
import numpy as np
from math import sqrt
import gurobipy
from gurobipy import GRB, Model, quicksum
import geopy.distance
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D     
import matplotlib.patches as mpatches
import tkinter as tk
from tkinter import ttk
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def geodesic(lat1, lon1, lat2, lon2): # Haversine Formula Function
    R = 6371  # Earth's radius in kilometers

    # Convert decimal degrees to radians
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    # Apply Haversine Formula
    a = math.sin(dLat / 2)**2 + math.sin(dLon / 2)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance


df = pd.read_excel("Test Data.xlsx",index_col=0) # Reading the dataset

xcoord = list(df['Latitude']) # Reading X-coordinate from the dataset
ycoord = list(df['Longitude']) # Reading Y-coordinate from the dataset
q = list(df['demand']) # Reading Demand from the dataset
e = list(df['Earliest']) # Reading Earliest start time from the dataset
l = list(df['Latest']) # Reading Latest start time from the dataset
s = list(df['Service Time']) # Reading Service time from the dataset

# Haversine distance calculation, Dij parameter
C_geo = [[geodesic(xcoord[i],ycoord[i],xcoord[j],ycoord[j]) for j in range(len(xcoord))] for i in range(len(xcoord))]

# Convert C_geo to a NumPy array
C_geo = np.array(C_geo)

Q = 202.4  # Vehicle capacity, capacity of the truck is 9200 kg and there are 22 working days in a month which equals to 9200 * 22 = 202400

Q_c = 118.8 # Backhaul cluster capacity, capacity of the backhaul cluster is 5400 kg and there are 22 working days in a month which equals to 5400 * 22 = 118800

F = 1795 # Average cost of each vehicle, 20,000 tl (fuel cost) + 25,000 tl (spare parts repair and maintenance cost) + 5000 tl (insurance cost) + 3000 tl (highway fees) + 500 tl (communication fees) = 53,500 tl total cost which is equal to 1,795 USD

A = 660 # Legal working hour limitation of the vehicle drivers which is 270 minutes (4.5 hour) per day

# Vehicle speed
vehicle_speed = 60  # 60 km per average in the local delivery
   
# Defining the sets
location_set = 51  # Include all locations i and j, total 51 locations, 1 depot, 44 delivery locations and65 pickup locations
clusters_set = 2  #There will be 1 clusters which are the pickup points, I will use the range as for c in range(1,clusters_set) that equals to 1
vehicle_set = 10  # There will be 9 vehicles, I will use the range as for k in range (1,vehicle_set) that equals to 9

M = 1e10  #Big M


# Tij parameter calculation, distance / vehicle speed (60 km/h)
t = [[C_geo[i][j] / vehicle_speed for j in range(location_set)] for i in range(location_set)] 

# Convert t to a NumPy array
t = np.array(t)

# Defining the model
VRPTWBC = gurobipy.Model()

# Enforcing the optimality gap
VRPTWBC.setParam('MIPGap', 0.01)

#VRPTWBC.setParam('TimeLimit', 60)

# Defining the decision variables
x = VRPTWBC.addVars(location_set, location_set, vehicle_set, lb=0, ub=1, vtype=GRB.BINARY, name='X') # 1 if arc (i, j) is traversed by a vehicle k; 0 otherwise
b = VRPTWBC.addVars(location_set, lb = 0, vtype = GRB.CONTINUOUS,name = 'B') # Service start time of node i by one of the vehicles
u = VRPTWBC.addVars(location_set, vehicle_set, lb = 0, vtype = GRB.CONTINUOUS,name = 'U') # Load on the vehicle k upon its arrival at node i 
y = VRPTWBC.addVars(clusters_set, vehicle_set, lb=0, ub=1, vtype=GRB.BINARY, name='Y') # 1 if backhaul cluster c is visited by vehicle k; 0 otherwise 
z = VRPTWBC.addVars(vehicle_set, lb=0, vtype=GRB.CONTINUOUS, name='Z') # Total time spent by each vehicle k



# Defining the objective function, goal is minimizing the total distance traveled and minimizing the total cost
VRPTWBC.setObjective(quicksum(C_geo[i, j] * x[i, j, k] for i in range(location_set) for j in range(location_set) if i != j for k in range(1, vehicle_set)) + F * quicksum(x[0, i, k] for i in range(location_set) for k in range(1, vehicle_set)), GRB.MINIMIZE)


#Adding the constraints

# Each customer is visited exactly once by one vehicle constraints. These are the flow balance constraints
VRPTWBC.addConstrs(quicksum(x[i,j,k] for j in range (location_set) for k in range (1,vehicle_set) if i != j) == 1 for i in range(1,location_set)) 

VRPTWBC.addConstrs(quicksum(x[i,j,k] for i in range (location_set) for k in range (1,vehicle_set) if i != j) == 1 for j in range(1,location_set)) 


# Starting from the depot constraint
VRPTWBC.addConstrs(quicksum(x[0, j, k] for j in range(1, location_set) if j != 0) == 1 for k in range(1, vehicle_set))

# Returning to the depot constraint
VRPTWBC.addConstrs(quicksum(x[i, 0, k] for i in range(1, location_set) if i != 0) == 1 for k in range(1, vehicle_set))

# Flow constraints to ensure each vehicle completes a single tour
VRPTWBC.addConstrs(quicksum(x[i, j, k] for j in range(location_set) if i != j) - quicksum(x[j, i, k] for j in range(location_set) if i != j) == 0 for i in range(1, location_set - 1) for k in range(1, vehicle_set))


# Time window constraints
VRPTWBC.addConstrs((b[i] >= e[i] for i in range(1, location_set)))

VRPTWBC.addConstrs((b[i] <= l[i] for i in range(1, location_set)))

VRPTWBC.addConstrs(b[i] + s[i] + t[i][j] - M * (1 - x[i,j,k]) <= b[j] for i in range(location_set) for j in range(1,location_set) for k in range (1,vehicle_set))


# Vehicle capacity and ensuring the demand of the customers constraints
VRPTWBC.addConstrs(u[j, k] >= q[j] for j in range (location_set) for k in range (1,vehicle_set)) 

VRPTWBC.addConstrs(u[j, k] <= Q for j in range (location_set) for k in range (1,vehicle_set))    

VRPTWBC.addConstrs(u[j,k] <= (u[i,k] - (q[i] * x[i,j,k]) + (Q * (1 - x[i,j,k]))) for i in range (location_set) for j in range (1, location_set) for k in range (1,vehicle_set) if i!=j)

VRPTWBC.addConstrs((u[0, k] >= 0 for k in range (1,vehicle_set))) 

VRPTWBC.addConstrs((u[0, k] <= Q for k in range (1,vehicle_set))) 


# This constraint ensures that each backhaul cluster is associated with a specific pickup point
VRPTWBC.addConstrs(quicksum(x[i, j, k] for i in range(1, location_set) for j in range(1, location_set) if i != j and q[i] < 0) == y[c, k]
                    for c in range(1, clusters_set) for k in range(1, vehicle_set))

# This constraint ensures that if a backhaul cluster is visited, the corresponding pickup point is visited
VRPTWBC.addConstrs(y[c, k] <= quicksum(x[i, j, k] for i in range(1, location_set) for j in range(1, location_set) if i != j and q[i] < 0)
                    for c in range(1, clusters_set) for k in range(1, vehicle_set))

# Each backhaul cluster is visited exactly once by a single vehicle
VRPTWBC.addConstrs(quicksum(y[c, k] for k in range(1, vehicle_set)) == 1 for c in range(1, clusters_set)) 

# This constraint ensures that total load at each backhaul cluster does not exceed its capacity
VRPTWBC.addConstrs(quicksum(q[i] * x[i, j, k] for i in range(1, location_set) for j in range(1, location_set) if i != j and q[i] < 0) <= Q_c * y[c, k]
                    for c in range(1,clusters_set) for k in range(1, vehicle_set))

# This constraint ensures that deliveries are done before pickups
for k in range(1, vehicle_set):
    for i in range(1, location_set):
        for j in range(1, location_set):
            if q[i] < 0 and q[j] > 0:
                # If i is a pickup location and j is a delivery location, enforce the order
                VRPTWBC.addConstr(x[i, j, k] == 0)


# This constraint calculates the total time spent by each vehicle k on its tour. It considers the sum of travel time, service time and job starting time for each visited location
VRPTWBC.addConstrs(z[k] == quicksum((t[i][j] + s[j]) * x[i, j, k] for i in range(location_set) for j in range(location_set) if i != j) for k in range(1, vehicle_set))

# This constraint ensures that the total time spent by each vehicle does not exceed the maximum allowed working time
VRPTWBC.addConstrs(z[k] <= A for k in range(1, vehicle_set))


# This constraint ensures that, when a pickup location is visited it alllows the demand to be loaded into the vehicle
VRPTWBC.addConstrs((u[j, k] == u[i, k] + q[j] * x[i, j, k] for i in range(location_set) for j in range(1, location_set) if q[j] < 0 and i != j and k in range(1, vehicle_set)))


# Solving the model
VRPTWBC.update()


# Optimize the model
VRPTWBC.optimize()



status = VRPTWBC.status
object_Value = VRPTWBC.objVal
print()
print("Model Status is: ", status)
print()
print("Objective Function Value is: ", object_Value)
print()
total_distance = VRPTWBC.objVal - F * sum(x[0, i, k].X for i in range(1, location_set) for k in range(1, vehicle_set))
print("Total Distance Traveled by the Mathematical Formulation is: ", total_distance, "kilometers")
print()
print("Value of the Decision Variables are: ")
print()

            
# Extract the optimal values of the decision variables
optimal_x_values = {(i, j, k): x[i, j, k].x for i in range(location_set) for j in range(location_set) for k in range(1, vehicle_set)}

# Print decision variables which are not zeros, and necessary U values
if status != 3 and status != 4:
    for v in VRPTWBC.getVars():
        if VRPTWBC.objVal < 1e+99 and v.x != 0:
            if v.Varname[0] == 'U' and v.x > 0.5:
                parts = v.Varname.split('[')
                i = int(parts[1].split(',')[0])
                k = int(parts[1].split(',')[1][:-1])
                if k in range(1, vehicle_set) and any(optimal_x_values.get((i, j, k), 0) > 0.5 for j in range(location_set)):
                    print('%s %f' % (v.Varname, v.x))
            else:
                print('%s %f' % (v.Varname, v.x))
           


print("\nEstablished Tours:")
for k in range(1, vehicle_set):
    print(f"\nVehicle {k} Tour: 0", end=" ")  # Start with location 0
    current_location = 0  # Starting from the depot
    while True:
        next_location = None
        for j in range(location_set):
            if optimal_x_values.get((current_location, j, k), 0) > 0.5:
                next_location = j
                break
        if next_location is None or next_location == 0:
            break  # End of the tour
        print(f" -> {next_location} ({'Delivery Location' if q[next_location] > 0 else 'Pickup Location'})", end=" ")  # Print the rest of the tour with labels
        current_location = next_location
    print(" -> 0")  # Add location 0 at the end of each vehicle's tour 

# For visualization part
xcoord = list(df['Latitude'])  # Reading X-coordinate from the dataset
ycoord = list(df['Longitude'])  # Reading Y-coordinate from the dataset

nodes = list(range(df.shape[0]))
arcs = []
vehicle_colors = {}  # To store the assigned color for each vehicle

# Use the 'tab10' colormap for more distinctive and prominent colors
for k in range(1, vehicle_set):
    color = plt.cm.tab10(k - 1)  # Use tab10 colormap for distinct colors
    vehicle_colors[k] = color
    for i in range(location_set):
        for j in range(location_set):
            if x[i, j, k].X == 1:
                arcs.append((i, j, k, color))

plt.figure(figsize=(15, 10))  # Size has been selected for the visualization figure

for i in arcs:
    plt.plot([xcoord[i[0]], xcoord[i[1]]], [ycoord[i[0]], ycoord[i[1]]], c=i[3])  # Arcs have been plotted with assigned colors
    if i[0] != 0:
        plt.text(xcoord[i[0]], ycoord[i[0]], i[0], fontdict=dict(color='black', alpha=0.5, size=16))

plt.text(xcoord[0], ycoord[0], 'depot', fontdict=dict(color='black', alpha=0.5, size=16))
plt.plot(xcoord[0], ycoord[0], c="r", marker='s')
plt.scatter(xcoord[1:], ycoord[1:], c="b")  # Coordinates x and y of the customer nodes have marked with the blue color
plt.title("Mathematical Formulation Solution")

# Add legend for vehicle colors
legend_handles = [mpatches.Patch(color=color, label=f'Vehicle {k}') for k, color in vehicle_colors.items()]
plt.legend(handles=legend_handles, loc='lower right')

plt.show()


# 2-opt Part of the Code
    
# Extract the optimal values of the decision variables for tours
optimal_tour_values = {(i, j, k): x[i, j, k].x for i in range(location_set) for j in range(location_set) for k in range(1, vehicle_set)}

# Create initial tours based on optimal values
tours = {}
for k in range(1, vehicle_set):
    current_location = 0  # Starting from the depot
    tour = [current_location]
    while True:
        next_location = None
        for j in range(location_set):
            if optimal_tour_values.get((current_location, j, k), 0) > 0.5:
                next_location = j
                break
        if next_location is None or next_location == 0:
            # Add the depot back to the tour to complete the loop
            tour.append(0)
            break  # End of the tour
        tour.append(next_location)
        current_location = next_location
    tours[k] = tour
    
    
# 2-opt Algorithm Part
def calculate_total_distance(route, d):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += d[int(route[i])][int(route[i + 1])]
    return total_distance

def two_opt(route, d):
    best_distance = calculate_total_distance(route, d)
    while True:
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:]
                new_route[i:j] = route[j - 1:i - 1:-1]
                new_distance = calculate_total_distance(new_route, d)
                if new_distance < best_distance:
                    route = new_route
                    best_distance = new_distance      
                    break  # improvement found, restart the outer loop
            else:
                continue  # no improvement found, continue to next pair
            break  # improvement found, break the inner loop
        else:
            break  # no improvement found, break the outer loop
    return route

# Apply 2-opt optimization to each tour
for t in tours:
    tours[t] = two_opt(tours[t], C_geo)
    
print()
print("Established Tours After Applying the 2-opt Algorithm:")
print()
# Print the final tours
for t in tours:
    print("Tour:",tours[t])
    
arcs = []
for t in tours:
    for index in range(len(tours[t])-1):
        arcs.append((int(tours[t][index]), int(tours[t][index+1])))

# Compute the total traveling distance
vrp_solution_length = 0    # Total length initially determined as 0 
for tour in tours.values():   # For all tour lists, we used .values() and we reached to the list
    for i in range(len(tour) - 1):        #We included the one previous index of the tour
        vrp_solution_length += C_geo[int(tour[i])][int(tour[i + 1])]     # We calculated the distances between the customer nodes in that tour, and we added to the vrp_solution_length 

# Round the result to 2 decimals to avoid floating point representation errors, it is not really necessary, because excel file contains flat values but with doing this we can ensure it
vrp_solution_length = round(vrp_solution_length, 2)

print()
print('With a Total Traveling Distance After Applying the 2-opt Algorithm: ', vrp_solution_length, "kilometers")
print()
print("Total Distance Saved by 2-opt Algorithm is: ", total_distance - vrp_solution_length, "kilometers")
print()

# Visualize the tours after 2-opt algorithm
plt.figure(figsize=(15, 10))

# Plot edges
for i in arcs:
    if len(i) == 4:  # Ensure that the tuple has the expected format
        plt.plot([xcoord[i[0]], xcoord[i[1]]], [ycoord[i[0]], ycoord[i[1]]], c=i[3])

# Plot tours
for t in tours:
    if len(tours[t]) > 1:
        for i in range(len(tours[t]) - 1):
            plt.plot([xcoord[tours[t][i]], xcoord[tours[t][i + 1]]], [ycoord[tours[t][i]], ycoord[tours[t][i + 1]]], c=vehicle_colors[t])

# Scatter plot for locations
plt.scatter(xcoord[1:], ycoord[1:], c="b")

# Text annotations for customer numbers with the same font size as the pre-2-opt visualization
for i in range(1, len(xcoord)):
    plt.text(xcoord[i], ycoord[i], f'{i}', fontsize=16, ha='center', va='bottom', color='black', alpha=0.5)

plt.text(xcoord[0], ycoord[0], 'depot', fontdict=dict(color='black', alpha=0.5, size=16))
plt.plot(xcoord[0], ycoord[0], c="r", marker='s')

# Add legend for vehicle colors
legend_handles = [mpatches.Patch(color=color, label=f'Vehicle {k}') for k, color in vehicle_colors.items()]
plt.legend(handles=legend_handles, loc='lower right')

plt.title("Established Tours After 2-opt Algorithm")
plt.show()



# Interface part of the code


# Update details and visualization functions
def update_details():
    # Clear the current details only
    details_text.delete("1.0", tk.END)
    update_button['state'] = tk.DISABLED
    vehicle_colors = []  # To store the assigned color for each vehicle

    # Use the 'tab10' colormap for more distinctive and prominent colors
    for k in range(1, vehicle_set):
        color = plt.cm.tab10(k - 1)  # Use tab10 colormap for distinct colors
        vehicle_colors.append(color)
        for i in range(location_set):
            for j in range(location_set):
                if x[i, j, k].X == 1:
                    arcs.append((i, j, k, color))

    # Iterate over vehicles and update routes
    for k in range(1, vehicle_set):
        current_location = 0  # Starting from the depot
        load = 202.4  # Initial load set to vehicle capacity

        ax.set_title("Interface Visualization")
        ax.scatter(xcoord[1:], ycoord[1:], c="b")  # Customer nodes
        ax.scatter(xcoord[0], ycoord[0], c="r", marker='s')  # Depot node

        # Call the function to update the route for the current vehicle
        current_location, load = update_vehicle_route(k, current_location, load, vehicle_colors)

# Function to update vehicle route
def update_vehicle_route(k, current_location, load, vehicle_colors):
    details_text.insert(tk.END, f"Vehicle {k} Route:\n")

    # Plot a line connecting the depot to the first location
    ax.plot([xcoord[0], xcoord[current_location]],
            [ycoord[0], ycoord[current_location]], c='black', linestyle='dashed', marker='o')

    visited_locations = {0}  # Start with the depot

    while True:
        next_location = None
        for j in range(location_set):
            if optimal_x_values.get((current_location, j, k), 0) > 0.5:
                next_location = j
                break

        if next_location is None or next_location in visited_locations:
            break  # End of the tour or already visited

        print(f"Current Location: {current_location}, Next Location: {next_location}")

        # Display details in the UI
        details_text.insert(tk.END, f"\nLocation: {next_location}\n")
        # Check if it's a delivery or pickup point
        if q[next_location] > 0:  # Delivery point
            delivered_demand = min(load, q[next_location])
            load -= delivered_demand
            details_text.insert(tk.END, f"Delivered Demand: {delivered_demand:.2f}\n")
        elif q[next_location] < 0:  # Pickup point
            # Adjust load for the first pickup location in the tour
            if current_location == 0:
                load = 0  # Assuming load as 0 for the first pickup location
            picked_up_demand = min(202.4 - load, -q[next_location])
            load += picked_up_demand
            details_text.insert(tk.END, f"Picked Up Demand: {picked_up_demand:.2f}\n")

        details_text.insert(tk.END, f"Load: {load:.2f}\n")
        details_text.insert(tk.END, f"Capacity: {202.4}\n")

        if current_location != 0:
            details_text.insert(tk.END, f"Previous Destination: {current_location}\n")

        # Plot the route with different colors for each vehicle
        color = vehicle_colors[k - 1]
        ax.plot([xcoord[current_location], xcoord[next_location]],
                [ycoord[current_location], ycoord[next_location]], c=color, marker='o')

        # Update variables for the next iteration
        current_location = next_location
        visited_locations.add(current_location)

        # Display the plot for the current vehicle after each iteration
        canvas.draw()
        root.update()  # Update the Tkinter window

        # Introduce a delay (you can adjust the duration as needed)
        root.after(1000)  # 1000 milliseconds = 1 second

    # Plot a line connecting the final visited location to the depot
    ax.plot([xcoord[current_location], xcoord[0]],
            [ycoord[current_location], ycoord[0]], c=color, linestyle='dashed', marker='o')

    details_text.insert(tk.END, "\n" + "=" * 30 + "\n")

    return current_location, load

# Create the main window
root = tk.Tk()
root.title("Vehicle Routing Problem with Time Windows and Backhaul Clusters")

# Set the size of the main window
root.geometry("1400x900")  # Adjust the dimensions as needed

# Create a Text widget to display details
details_text = tk.Text(root, height=30, width=80)
details_text.pack(padx=20, pady=20)

# Create a button to trigger details update
update_button = ttk.Button(root, text="Start Routing", command=update_details)
update_button.pack(pady=10)

# Create a Matplotlib Figure and Axes for real-time plot
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the dimensions as needed
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Scatter plot for locations with labels
for i in range(1, len(xcoord)):
    plt.scatter(xcoord[i], ycoord[i], c="b")
    plt.text(xcoord[i], ycoord[i], str(i), fontsize=10, ha='center', va='bottom', color='black', alpha=0.7)

# Text label for the depot
plt.text(xcoord[0], ycoord[0], 'Depot', fontsize=10, ha='center', va='bottom', color='black', alpha=0.7)

# Add legend for vehicle colors
legend_handles = [mpatches.Patch(color=color, label=f'Vehicle {k}') for k, color in vehicle_colors.items()]
plt.legend(handles=legend_handles, loc='lower right')

# Run the Tkinter main loop
root.mainloop()

# Vehicle-Routing-Optimisation-with-Backhaul-Clusters-Project

- Developed a vehicle routing optimisation with backhaul operations to model urban logistics (European side of Istanbul) under realistic operational constraints, including delivery and pickup locations, vehicle capacities, time windows, and working time limits.
  
- Mixed integer linear programming model was established by identifying the sets, parameters, decision variables, objective function and constraints with a goal of minimizing the total distance traveled and the total cost.

- Developed constraints ensured that each customer is visited exactly once, each route starts and ends at the depot, flow/balance, time window requirements are met, vehicle capacity limitation is enforced, customer demands are satisfied, backhaul cluster requirements are fulfilled, deliveries are done before pickups, vehicle loads are updated after pickup and delivery operations, and legal working hours of each vehicle are not exceeded.

- The mathematical model was coded using Python, Gurobi was used as the optimisation solver tool and established routes were visualized.

- 2-opt Algorithm, a local search algorithm, was implemented for route optimization (distance reduction) with a goal of enhancing the efficiency of the initial solution.

- Tkinter-based user interface for real-time visualization was developed, allowing users to observe the routing process interactively. Next destination, previous destination, vehicle capacity, load on the vehicle, and delivered/picked-up demand values were provided in the Tkinter-based user interface to the users.

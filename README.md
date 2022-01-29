# SimulatePandemic

Simple program for simulating pandemic disease spread via a global airplane traffic network. Each node represents a city with an airport, and the edges represent airline traffic connections between airports. The spread of the disease is simulated using a simple SIR model, which is based on three coupled differential equations of the number of susceptible (S), infected (I), and recovered (R) people in each node of the network over time.

<img src="https://render.githubusercontent.com/render/math?math=\frac{dS}{dt} = -\frac{\beta IS}{N}, ">
<img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt} = -\frac{\beta IS}{N} - \gamma I, ">
<img src="https://render.githubusercontent.com/render/math?math=\frac{dR}{dt} = \gamma I. ">

The simulation program can be extended to include more types of transportation networks, and to import realistic travel flux between the nodes to produce a more realistic simulaiton.

![simulation](https://user-images.githubusercontent.com/24233036/151661006-88a58d5e-6b32-4576-a4fd-4e49affceb5f.png)

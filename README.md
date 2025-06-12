# Project Details

This is a reupload of my private (and admittedly quite messy) repository.

This project consists of two main components:  
1. A **light intensity simulation** that calculates the intensity of light at specific points on a plane.  
2. A **path reconstruction algorithm** that attempts to infer the path of a moving object using limited information.

---

## Light Intensity Component

This component performs rigorous, physics-based calculations to determine light intensities at various points in space. Unlike methods optimized for speed—such as those used in graphics rendering—this simulation prioritizes mathematical accuracy and physical correctness. As such, it is more suitable for analysis where precise measurements are critical, such as in simulations or technical validation.

Due to the computational complexity, the simulation is not optimized for real-time performance. It typically takes ~3-5 minutes to run and may cause some lag. Feel free to grab a cup of coffee (or water) while you wait!

---

## Path Reconstruction Algorithm

The primary focus of this repository is the path reconstruction algorithm.

In this simulation, a virtual "rover" moves across a two-dimensional plane (referred to as the *floor*), recording the light intensity at its location at a specified sampling rate (default is every 0.2 units). The rover's path is not directly observable. However, given:

- Its initial position  
- The sampling rate  
- The sequence of recorded intensity values  
- A precomputed intensity map of the floor  

The algorithm attempts to infer the path the rover has taken.

This type of algorithm is inspired by and applicable to problems in the field of **Visible Light Communication (VLC)**, particularly in areas involving localization and tracking of devices based on light intensity patterns. It can serve as a foundation for further research into passive tracking, indoor positioning, and similar applications in VLC systems.

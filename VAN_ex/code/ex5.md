# EX4


## 5.1 
- The extrinsic camera matrix $[R|t]$, the transformation from camera's coordinate system to world coordinate system:

$$
\begin{align}
    [R^T|-R^Tt]
\end{align}
$$

- Picking a random track of length $\ge 10$, here is the reprojection error graph ($L_2$ norm):
[![reprojection error](../outputs/ex5/reproj_err.png "track reprojection error")](../outputs/ex5/reproj_err.png)


- The following graph presents the factor error over the track's frames: [![factor error before optimization](../outputs/ex5/factor_error_before_optimization.png "factor_error_before_optimization")](../outputs/ex5/factor_error_before_optimization.png)

As expected, the factor error error increases as the frame is farther apart from the last frame, which is the frame we set as constraint.

- The following graph presents the factor error as a function of the reprojection error (before optimization). As seen in the graph, they are directly proportional to each other. [![factor_error_vs_reproj_err_before_optimization](../outputs/ex5/factor_error_vs_reproj_err_before_optimization.png "factor_error_vs_reproj_err_before_optimization")](../outputs/ex5/factor_error_vs_reproj_err_before_optimization.png)

## 5.2
Choosing the key frames:
I chose key frames based on relative distance between frames. I chose the first frame as a key frame, and then I measured the relative distance maximums. The first and last frames were also chosen as key frames.
Here is a graph of the selected key frames:
[![key frames](../outputs/ex5/key_frames.png "key frames")](../outputs/ex5/key_frames.png)

One of the problems optimizing the graph is outliers. Since the graph attempts to optimize all factors, outliers can have a big impact on the optimization. To avoid this, I used the robust kernel function to the factors. I used the Cauchy kernel function:
```
measurement_noise = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy(1.0), projection_uncertainty)
```

Tha cauchy kernel function can be seen here:
The influence of outliers is reduced, as the residues grow.
[![cauchy kernel](../outputs/ex5/cauchy_kernel.png "cauchy kernel")](../outputs/ex5/cauchy_kernel.png)

The factor graph error of first bundle window: **before** optimization: 

| before optimization | after optimization |
|---------------------|--------------------|
|  34.14038161373287  | 31.781447534814216 |


3D plots of the camera positions before and after optimization:
[![camera positions before optimization](../outputs/ex5/initial_camera_poses_first_bundle.png "camera positions before optimization")](../outputs/ex5/initial_camera_poses_first_bundle.png)
[![camera positions after optimization](../outputs/ex5/result_camera_poses_first_bundle.png "camera positions after optimization")](../outputs/ex5/result_camera_poses_first_bundle.png)

Plots of camera position with landmarks as seen from above:
[![first bundle window as seen from above](../outputs/ex5/result_trajectory_and_points_first_bundle.png "first bundle window as seen from above")](../outputs/ex5/result_trajectory_and_points_first_bundle.png)

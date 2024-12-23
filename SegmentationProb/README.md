# Probabilistic Data Segmentation for Imitation Learning

This document describes the complete process of the algorithm presented for data segmentation and change point detection. This approach uses a combination of finite differences, jerk calculation (the third derivative of position), moving averages, and probabilistic modeling to identify key change points in the data. Each step is explained mathematically below.

## 1. Finite Differences

The algorithm is based on the numerical approximation of derivatives to calculate the jerk (third derivative of position with respect to time). The general formula for calculating the derivative of a function $ f(x)$ discretely is:

$f'(x_i) \approx \frac{f(x_{i+1}) - f(x_i)}{t_{i+1} - t_i}$

In this case, if $f(x)$ is the position of a robot over time, the first derivative will be the velocity $v(t)$, the second derivative will be the acceleration $a(t)$, and the third derivative will be the jerk $j(t)$.

The calculation of the derivatives is performed using the `calc_time_deriv` function as follows:

$\text{deriv}_i = \frac{\text{data}_{i+1} - \text{data}_i}{\text{time}_{i+1} - \text{time}_i}$

This process is repeated three times to obtain the jerk.

## 2. Jerk Calculation

Jerk is the third derivative of position with respect to time, defined as:

$j(t) = \frac{d^3}{dt^3} p(t)$

where $p(t)$ is the position of the robot as a function of time. In the algorithm, this value is calculated using the `calc_jerk_in_time` function, which applies three consecutive derivatives to the position data.

## 3. Moving Average

After calculating the jerk, a moving average is applied to smooth the signal. The moving average of a signal $f$ over a window of size $w$ is defined as:

$\text{Ma}(f)_i = \frac{1}{w} \sum_{k=0}^{w-1} f_{i+k}$

The algorithm uses the `moving_average` function for this purpose, reducing noise in the signal before proceeding with change point detection.

## 4. Normalization

After calculating the moving average of the jerk, the resulting signal is normalized by dividing it by its maximum value:

\[
\text{norm\_avg\_jerk}(t) = \frac{\text{avg\_jerk}(t)}{\max(\text{avg\_jerk})}
\]

This normalization ensures that the values are kept within a standardized range (0 to 1), facilitating change point detection based on a threshold.

## 5. Change Point Detection

A threshold method is implemented to detect change points. The algorithm looks for points where the normalized signal exceeds a predefined threshold. The detection criterion is:

$\text{seg}(i) = 
\begin{cases} 
1, & \text{if} \ \text{norm\_avg\_jerk}(i) \geq \text{threshold} \ \text{and the segment size is greater than the minimum}\\
0, & \text{otherwise}
\end{cases}$

The minimum number of points that the signal must exceed the threshold to be considered a change is controlled by the `segment_size` parameter. The `count_thresh` function performs this operation.

## 6. Probabilistic Modeling of Change Points

Once the segments are detected, the algorithm probabilistically models each change point as a Gaussian distribution. The probability of a change point is modeled as a Gaussian density function:

$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$

where $\mu$ is the mean (the change point) and $\sigma$ is the standard deviation, which is related to the size of the smoothing window.

For each change point $p_i$, the probability of its occurrence is calculated as:

$\text{p}(x) = \sum_{i=1}^{N} \mathcal{N}(x; p_i, \sigma^2)$

This operation is performed in the `calc_segment_prob` function.

## 7. Probabilistic Combination and Sampling

Finally, the change points from multiple signals are probabilistically combined (if necessary) and weighted sampling is performed to determine the most probable key points. The combined probability is obtained by multiplying the individual probabilities of each signal:

$P_{\text{total}}(x) = P_1(x) \times P_2(x) \times \dots \times P_N(x)$

Then, weighted sampling is performed to select the most probable key points. This is done with the `probabilistically_combine` function.

## 8. Visual Representation

The algorithm has the ability to generate visual representations of the segments and key points using 2D and 3D plots. This is done using the `matplotlib` and `mplot3d` libraries. The identified segments are shown in different colors for better visual understanding.

## 9. Implementation in 2D and 3D

The algorithm can be applied to both 2D and 3D trajectories. In the 2D example (`main2d`), a `.h5` file containing data from a two-dimensional trajectory is used. In the 3D example (`main3d`), the algorithm is applied to position data of a robot, processing signals such as joint positions, forces, and gripper position.

---

This algorithm is useful for detecting key points in robotic movements and can be easily adapted to different types of signals and data by adjusting the segmentation, threshold, and probabilistic modeling parameters.


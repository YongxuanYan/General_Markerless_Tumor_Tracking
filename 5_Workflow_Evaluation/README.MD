为了评估肿瘤中心点在3维空间中的运动轨迹误差，我们需要确定来自grab0和grab1两组立体透视图像中在一组图像序列中选定的基准标记点在另一组中是哪一个。

为了实现基准标记点的确定，我们利用CT数据，在CT中选定合适的基准标记，然后将基准标记周围半径5mm的球状区域的CT值设置为3000. 
由此生成的DRR就可以在DRR中看到一个硬币形状的区域，其中心点就是基准标记所在的位置，我们通过将生成DRR的三维空间位置信息与实际采集立体透视图像时的信息对齐，
生成相应的DRR，来确定选定的基准标记在两组透视图像中的位置。

实现过程：首先读取CT数据，并保存为.npy格式，然后新建python项目，将.npy数据放到项目根目录，在项目根目录中放入set_fiducial_marker_area_obvious.py，
运行该.py文件后按照提示通过点击图像中的目标点来选定基准标记点，点击apply以后，会将该点周围5mm半径的球状区域的CT值设置为3000，然后点击保存即可覆盖原来的.npy数据。

然后使用该.npy数据生成DRR图像，再对照CFI图像对，并做好标记与记录。

确定好选定的基准标记在所有的图像序列中的位置以后，我们利用光流法对基准标记进行追踪，运行tack_fiducial_markers.py，在弹出的图像中选定追踪目标（记录的基准标记点），
追踪结果会保存到.../Trajectories/*.csv中，为了验证光流法追踪是否精准，该script会对每一个图像自动生成所追踪点的结果图，用绿色方框指示追踪结果，保存在.../Trajectories/Markers中

我们需要挨个检查每一个序列，对于失败的案例，请尝试更改追踪通道，将tack_fiducial_markers.py中的
r_channel = first_frame[:, :, 0]和curr_img = 255 - curr_frame[:, :, 0]
改为通道1或者2

确定所有目标点被正确追踪以后，就可以利用2组成像系统的三维空间位置信息，以及基准标记点和肿瘤中心点在图像中的运动轨迹。

Process_marker_trajectories_csv_files.py用于处理光流法生成的记录基准标记运动轨迹的csv文件，删除所有CSV文件中'patient','field'和'device'列，添加成像系统几何信息：'X tube x','X tube y','X tube z','IPEL'和'OID'列，便于我们使用compute_intersection.py进行批量计算

add_couch_angle_info.py用于添加每一个成像序列的治疗床角度信息。

split_full_trajectories.py用于将肿瘤中心点运动轨迹从Evaluate_DUCK_Marker_Track.py生成的full_trajectories.csv中拆分所有序列的数据到单独的csv文件中，并添加相应的几何信息，便于使用compute_intersection.py进行批量计算

compute_intersection.py用于根据成像系统几何信息以及基准标记点在立体透视图像中的位置，计算基准标记在三维空间中的运动轨迹。

compute_intersection_tumor_LPF_adaptive.py用于根据成像系统几何信息以及预测的肿瘤中心点在立体透视图像中的位置，计算肿瘤中心点在三维空间中的运动轨迹，与基准marker的3D运动轨迹计算不同的是，我们需要首先将肿瘤中心点的坐标与基准标记进行对齐，并通过分析覆盖90%情况下的marker最大频率（30采样率条件下2.2Hz），使用2.2Hz作为截止频率进行低通滤波，这既可以保留关键信息，又可以滤除抖动。

analyze_3D_trajectory_errors.py用于分析预测的肿瘤中心点在3维空间中的运动轨迹误差，并作图







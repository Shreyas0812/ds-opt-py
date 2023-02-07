# lpv_ds_python
The primary math and illustration can be found at: https://github.com/nbfigueroa/ds-opt. 

If you only need the clustering method, please follow the instruction here for phys_gmm_python package: https://github.com/penn-figueroa-lab/phys_gmm_python

For any questions concerning the usage please contact: HuiTakami@gmail.com. If you are one of the Lab Slack members, please dm me through Slack.

Note: Feb.6 Update: modify the code of lpv-ds optimization, now it could compile and optimze within 0.2 second.

# main function (open main.py file)
To use this code,  you should change this variable to your current package directory:
  ```
 pkg_dir = ‘.../ds-opt-python'
  ```
For Windows users, please reverse the ‘\‘ to ‘/’ and add an ‘r’ behind the directory:
  ```
 pkg_dir = r ‘...\ds-opt-python'
  ```
And also modify this code in  ```load_dataset_DS.py```
  ```
final_dir = pkg_dir + "/datasets/" + dataset_name
  ```
 to
  ```
final_dir = pkg_dir + r"\datasets\" + dataset_name
  ```


These are the parameters that you should modify for loading different trajectories and settings
  ```
chosen_dataset = 6  (The corresponding relationship could be found in  ```load_dataset_DS.py```)
sub_sample = 2  # '>2' for real 3D Datasets, '1' for 2D toy datasets
nb_trajectories = 4  # If you have multiple demonstrations in one dataset, choose the amount of them
 ```

# Additional Variables for Testing
```do_3d_plot = True``` 
If you are running a 3D dataset, you could set it to true to visualize the learning result. If you are running a 2D dataset, set it to False. The 2D function will be available in the next commitment. 

 ```save_gibbs_result = 1 ```
If 1, do Gibbs sampling and save the current result, if 0, skip Gibb sampling and load the recently saved result. 

 ```save_opt_result = 1 ``` 
if 1, do optimization to get A_k and b_k parameters and save them into two files ```A_k.npy``` and ```b_k.npy```. If 0, load two files we saved and convert the dynamical system into a yaml file named `haruhi’ and convert the npy files to ```A_k_b_k.mat``` for Matlab usage.  



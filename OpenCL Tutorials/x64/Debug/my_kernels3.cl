//minimum value reduce
__kernel void value_min(__global const int* A, __global int* B, __local int* min) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	min[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all threads to finish copying
	
	//cycles through each work group and places the lowest value at the front of the group
	for (int i = 1; i < N; i *= 2) {
		//calculates the minimum value of each work group
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if (min[lid] > min[lid + i]){
				min[lid] = min[lid + i];
			}
			
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Places the lowest values from each work group into an array, leaving the 0th element empty
	if (lid == 0)
		B[(id/N)+1] = min[lid];
			
		barrier(CLK_LOCAL_MEM_FENCE);
}

//maximum value reduce
__kernel void value_max(__global const int* A, __global int* B, __local int* max) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	max[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all threads to finish copying
	
	//cycles through each work group and places the highest value at the front of the group
	for (int i = 1; i < N; i *= 2) {
		//calculates the maximum value of each work group
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if (max[lid] < max[lid + i]){
				max[lid] = max[lid + i];
			}
			
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Places the lowest values from each work group into an array, leaving the 0th element empty
	if (lid == 0)
		B[(id/N)+1] = max[lid];
			
		barrier(CLK_LOCAL_MEM_FENCE);
}

//additive value reduce for average
__kernel void value_avg(__global const int* A, __global float* B, __local float* avg) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	avg[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE); //wait for all threads to finish copying
	
	//cycles through each work group, summing all of the values together and places the total value at the front of the group
	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			avg[lid] += avg[lid + i];
			
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Places the summed values from each work group into an array, leaving the 0th element empty
	if (lid == 0)
		B[(id/N)+1] = avg[lid];
			
		barrier(CLK_LOCAL_MEM_FENCE);
}

//
__kernel void value_hist(__global const int* A, __global int* H, int bins, int offset) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id]/10; //take value as a bin index

	//takes off the lowest negative value from all numbers so that all are above 0
	bin_index -= offset;

	if(bin_index < bins) //checks to make sure that bin_index does not exceed the amount of bins set for the histogram
		atomic_inc(&H[bin_index]); //serial operation, not very efficient!

	barrier(CLK_GLOBAL_MEM_FENCE);
}
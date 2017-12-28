function  [flow_magnitude ,vx ,vy] = VIF_create_frame_flow(Prev_F, Current_F,N , M)

	hight = size(Current_F,1);
	width = size(Current_F,2);

	B_hight = floor((hight - 11)/N);
	B_width = floor((width - 11)/M);

	alpha = 0.0026;
	ratio = 0.6;
	minWidth = 20;
	nOuterFPIterations = 7;
	nInnerFPIterations = 1;
	nSORIterations = 30;

	para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];
	% Coarse2FineTwoFrames function is optical flow implementation that can downloaded from http://people.csail.mit.edu/celiu/OpticalFlow/
	[vx,vy,warpI2] = Coarse2FineTwoFrames(double(Prev_F),double(Current_F),para);
	flow_magnitude = sqrt(vx.^2+vy.^2);

end




function feature_vec = VIF_create_feature_vec(path,file_name)
%       
% Inputs:
%          path , file_name    - of the AVI file. 
%
% Outputs:
%          feature_vec    -  vector of VIF features, size = M * N * 21. 

	FR   = 25;            % frame rate
	movment_int = 3;      % frames intervat between Current frame and Prev frame
	N = 4;                % number of  vertical blocks in frame
	M = 4;                % number of  horisontal blocks in frame

	K=4;

	mov = aviread(fullfile(path,file_name));


	frame_gap = 2*movment_int;

	index = 0;
	flow = zeros(100,134);
	% for every Frame
	for f = 1:frame_gap:length(mov)- frame_gap -5   
		
		% Ignore 3 first frames of the clip 
		Prev_F =           mov(f + 3).cdata;                                    
		Current_F =        mov(f + 3 + movment_int).cdata;
		Next_F =           mov(f + 3 + 2*movment_int).cdata;
		
		% if colored movie change to gray levels
		if size(Current_F,3)>1                                                       
			Prev_F = rgb2gray(Prev_F);
			Current_F = rgb2gray(Current_F);
			Next_F = rgb2gray(Next_F);
		end
		
		Prev_F = single(Prev_F);
		Current_F = single(Current_F);
		Next_F = single(Next_F);

		rescale = 100 / size(Current_F,1);
		if rescale < 0.8
			Prev_F = imresize(Prev_F, rescale);
			Current_F = imresize(Current_F, rescale);
			Next_F = imresize(Next_F, rescale);
		end

		[m1,vx1,vy1] = VIF_create_frame_flow(Prev_F, Current_F,  N, M);
		index = index + 1;
		[m2,vx2,vy2] = VIF_create_frame_flow(Current_F, Next_F,  N, M );
		delta = abs(m1 - m2);
		flow = flow + double(delta > mean2(delta));
	end
	flow = flow./index;
	feature_vec = VIF_create_block_hist(flow,N,M);

end

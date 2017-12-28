function frame_hist = VIF_create_block_hist( flow,N,M )

	hight = size(flow,1);
	width = size(flow,2);

	B_hight = floor((hight - 11)/N);
	B_width = floor((width - 11)/M);

	frame_hist = [];
	for y = 6: B_hight:hight - B_hight - 5
		for x = 6: B_width:width - B_width - 5
			block_hist = VIF_block_hist(flow( y: y + B_hight -1, x: x + B_width -1,:));
			frame_hist = [frame_hist ; block_hist];
		end
	end

end

function  block_hist = VIF_block_hist(flow)

    flow_vec = reshape(flow, numel(flow), 1);
    Count = histc(flow_vec,0:0.05:1);
    block_hist = Count/sum(Count);

end




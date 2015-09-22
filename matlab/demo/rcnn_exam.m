addpath(genpath('../'));
addpath(genpath('../selective_search/'));

% caffe net
MODEL = '../../MODELs/bvlc_reference_caffenet/deploy.prototxt';
WEIGHTS='../../MODELs/bvlc_reference_caffenet/bvlc_reference_caffenet.caffeMODEL';
% rcnn
MODEL = '../../MODELs/bvlc_reference_rcnn_ilsvrc13/deploy.prototxt';
WEIGHTS='../../MODELs/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffeMODEL';
% load mean file
%mean_data = caffe.io.read_mean('../../data/ilsvrc12/imagenet_mean.binaryproto');
d = load('../+caffe/imagenet/ilsvrc_2012_mean.mat');
mean_data = d.mean_data;
% load net
caffe.set_mode_cpu();
net = caffe.Net(MODEL, WEIGHTS, 'test');
% load synset_id
[synsets, tags] = textread('../../data/ilsvrc12/det_synset_words.txt', '%s %s');

for batch=1:50000
  try
		url = sprintf('../../images/fish-bike.jpg', batch);
    tic; im = imread(url); tocImread = toc;
		%image512 = imresize(im, [512 512], 'bicubic');
		image512 = im;

		% do selective_search
		boxes = floor(selective_search_boxes(image512, 1, 500));

		% draw bboxes
		hfig = figure('Visible', 'off'); image(uint8(image512)); axis('off');
		for bb=1:size(boxes, 1)
			bbox = boxes(bb,:);
			hold on; rectangle('Position', [bbox(2), bbox(1), bbox(4)-bbox(2), bbox(3)-bbox(1)], 'EdgeColor', 'g', 'LineWidth', 1);
		end
		saveas(hfig, sprintf('%08d_bboxes.png', batch));

		% get window images (+16pxl wrapping)
		region_proposals = GetRectsWithinImage(boxes, size(image512,1), size(image512,2), image512, 256);
		det_score = zeros(200, size(region_proposals,4));
		fprintf('total nb. of proposals: %d\n', size(region_proposals,4));
		for regionIdx = 1:size(region_proposals,4) 
			im_ = single(region_proposals(:,:,:,regionIdx));
			score = prediction(net, im_, mean_data);
			det_score(:,regionIdx) = score;
		end
		scores = det_score';
		bboxes = boxes(:, [2 1 4 3]);
		num_classes = 200;
		thresh = 0;
		dets = cell(num_classes, 1);
		for i = 1:num_classes
  			I = find(scores(:, i) > thresh);
  			scored_boxes = cat(2, bboxes(I, :), scores(I, i));
  			keep = nms(scored_boxes, 0.3);
  			dets{i} = scored_boxes(keep, :);
		end
		[s,id] = max(det_score,[], 2);

		col = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
		hfig = figure('Visible', 'off'); image(uint8(image512)); axis('square'); axis('off');
		for cand=1:num_classes
			if size(dets{cand},1) > 0
				bbox = dets{cand};
				bbox = bbox(:, [2 1 4 3 5]);
				num_boxes = size(bbox,1);
				for bb=1:num_boxes
					hold on; rectangle('Position', [bbox(bb,2), bbox(bb,1), bbox(bb,4)-bbox(bb,2), bbox(bb,3)-bbox(bb,1)], 'EdgeColor', col(2), 'LineWidth', 1);
					hold on; text(bbox(bb,2), bbox(bb,1), ['score: ', num2str(bbox(bb,5))], 'FontSize', 12);
				end
			end
		end
		saveas(hfig, sprintf('%08d_resize512.png', batch));
		fprintf('End of %d\n', batch);
    fprintf(  'ILSVRC2012_validation_%08d.JPEG %d %d %d %d %d %.4f %.4f %.4f %.4f %.4f (%.3f %.3f %.3f)\n', batch, h(1), h(2), h(3), h(4), h(5), p(1), p(2), p(3), p(4), p(5), tocImread, preprocessToc, toc);
  catch exception
    fprintf('%d, ERROR: %s\n', batch, exception.message);
    continue;
  end
end


function feature = asvspoof19_load_feature(feature_path, file_list, dataset)

if ~exist('dataset','var') 
    dataset = 'data';
end

if dataset(1) ~= '/'
    dataset = ['/', dataset];
end

show_message(feature_path);

feature = cell(size(file_list));
for i = 1 : length(file_list)
    filename = fullfile(feature_path, [file_list{i}, '.h5']);
    data = h5read(filename, dataset);
    datasize = size(data);
    if datasize(2) == 1
        data = data';
    end
%     if ndims(data) == 1
%         data = reshape(data, 1, length(data));
%         disp(file_list{i});
%     end
    feature{i} = double(data');
end

end
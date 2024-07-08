function clear_path(path_name)

if exist(path_name, 'dir')
    rmdir(path_name, 's');
end

mkdir(path_name);

end
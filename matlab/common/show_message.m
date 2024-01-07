function show_message(message)

if ~exist('message','var') 
    disp(datestr(now, '[yyyy-mm-dd HH:MM:SS]'));
else
    disp([datestr(now, '[yyyy-mm-dd HH:MM:SS]: '), char(message)]);
end


end
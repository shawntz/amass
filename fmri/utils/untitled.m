% Configuration
subjects = {'Y033', 'Y034'}; % <- List of subject strings like 'Y033', 'Y034'
condition_column = 'ret_trial_type';  % Choose the condition column
stim_fixed_duration = 4;  % Duration for last stim trial

for s = 1:length(subjects)
    subj_code = subjects{s};    % e.g., 'Y033'
    subj_num = str2double(subj_code(2:end)); 
    subj_out = sprintf('sub-%03d', subj_num); % 'sub-033'
    
    subj_folder = fullfile(subj_out, 'func');
    if ~exist(subj_folder, 'dir')
        mkdir(subj_folder);
    end
    
    cd(subj_folder);
    
    % List all run files
    files = dir(sprintf('amass_fmri_sub_%s_session_1_block_*_phase_ret_data.csv', subj_code));
    
    for f = 1:length(files)
        file_name = files(f).name;
        
        % Extract block number and increment for run number
        block_token = regexp(file_name, 'block_(\d+)', 'tokens');
        block_num = str2double(block_token{1});
        run_num = block_num + 1; 
        
        % Read the data
        data = readtable(file_name);
        
        % Calculate all durations
        n = height(data);
        data.pregoal_duration = data.goal_onset - data.pregoal_fixation_onset;
        data.goal_duration = data.preprobe_fixation_onset - data.goal_onset;
        data.preprobe_duration = data.stim_onset - data.preprobe_fixation_onset;
        
        stim_durations = NaN(n,1);
        for i = 1:n-1
            stim_durations(i) = data.pregoal_fixation_onset(i+1) - data.stim_onset(i);
        end
        stim_durations(n) = stim_fixed_duration;
        data.stim_duration = stim_durations;
        
        % Define onset-duration column pairs
        onset_columns = {'pregoal_fixation_onset', 'goal_onset', 'preprobe_fixation_onset', 'stim_onset'};
        duration_columns = {'pregoal_duration', 'goal_duration', 'preprobe_duration', 'stim_duration'};
        
        % Get unique condition values
        conditions = unique(data.(condition_column));
        
        % For each onset/duration pair
        for odx = 1:length(onset_columns)
            onset_col = onset_columns{odx};
            duration_col = duration_columns{odx};
            
            % Create folder if doesn't exist
            if ~exist(onset_col, 'dir')
                mkdir(onset_col);
            end
            
            % For each condition
            for c = 1:length(conditions)
                cond = conditions{c};
                cond_mask = strcmp(data.(condition_column), cond);
                
                onset_times = data.(onset_col)(cond_mask);
                durations = data.(duration_col)(cond_mask);
                
                if isempty(onset_times)
                    continue; % Skip if no trials
                end
                
                timing_matrix = [onset_times, durations];
                
                % Define output filename
                output_file = fullfile(onset_col, sprintf('%s_run-%02d.txt', cond, run_num));
                
                % Save the timing matrix
                save(output_file, 'timing_matrix', '-ascii');
            end
        end
    end
    
    cd ../..
end

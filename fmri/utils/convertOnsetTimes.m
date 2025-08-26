% %---------------------%
% % Convert Onset Times %
% %---------------------%
% 
% % Convert AMASS beh timing files into a two-column format that can
% % be read by SPM
% 
% % The columns are:
% % 1. Onset (in seconds); and
% % 2. Duration (in seconds
% 
% 

%% configuration
condition_columns = [  % condition(s) to split by
    'ret_trial_type', ...
    'stim_animacy', ...
    'enc_goal', ...
    'ret_goal', ...
    'ret_goal_type', ...
    'ret_label'
    ];

subjects = {'Y033'};  % subject ids to process

for s = 1:length(subjects)
    old_id   = subjects{s};
    subj_num = str2double(subj_id(2:end));
    subj_out = sprintf('sub-%03d', subj_id);
    subj_dir = fullfile(subj_out, 'func');

    cd(subj_dir);
    
    beh_files = dir(sprintf( ...
        'amass_fmri_sub_%s_session_1_block_*_phase_ret_data.csv', old_id ...
    ));

    for f = 1:length(beh_files)
        file_name   = beh_files(f).name;
        block_token = regexp(file_name, 'block_(\d+)', 'tokens');
        block_num   = str2double(block_token{1});
        run_num     = block_num + 1;
        data        = readtable(file_name);
        n_trials    = height(data);
        
        data.pregoal_duration = ...
            data.goal_onset - data.pregoal_fixation_onset;

        data.goal_duration = ...
            data.preprobe_fixation_onset - data.goal_onset;

        data.prestim_duration = ...
            data.stim_onset - data.preprobe_fixation_onset;

        stim_durations = NaN(n_trials, 1);

        for i = 1:n_trials - 1
            stim_durations(i) = data.pregoal_fixation_onset(i + 1) - ...
                data.stim_onset(i);
        end
        
        stim_durations(n)  = 4;  % fixed final trial duration at 4 seconds
        data.stim_duration = stim_durations;
    end
end





for subject=subjects
    
    subject = num2str(subject, '%02d'); % Zero-pads each number so that the subject ID is 2 characters long

    cd(['sub-' subject '/func']) % Navigate to the subject's directory

    Run1_onsetTimes = tdfread(['sub-' subject '_task-flanker_run-1_events.tsv'], '\t'); % Read onset times file
    Run1_onsetTimes.trial_type = string(Run1_onsetTimes.trial_type); % Convert char array to string array, to make logical comparisons easier

    Run1_Incongruent = [];
    Run1_Congruent = [];

    for i = 1:length(Run1_onsetTimes.onset)
        if strtrim(Run1_onsetTimes.trial_type(i,:)) == 'incongruent_correct'
            Run1_Incongruent = [Run1_Incongruent; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        elseif strtrim(Run1_onsetTimes.trial_type(i,:)) == 'incongruent_incorrect'
            Run1_Incongruent = [Run1_Incongruent; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        elseif strtrim(Run1_onsetTimes.trial_type(i,:)) == 'congruent_correct'
            Run1_Congruent = [Run1_Congruent; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        elseif strtrim(Run1_onsetTimes.trial_type(i,:)) == 'congruent_incorrect'
            Run1_Congruent = [Run1_Congruent; Run1_onsetTimes.onset(i,:) Run1_onsetTimes.duration(i,:)];
        end
    end

    Run2_onsetTimes = tdfread(['sub-' subject '_task-flanker_run-2_events.tsv'], '\t');
    Run2_onsetTimes.trial_type = string(Run2_onsetTimes.trial_type);

    Run2_Incongruent = [];
    Run2_Congruent = [];

    for i = 1:length(Run2_onsetTimes.onset)
        if strtrim(Run2_onsetTimes.trial_type(i,:)) == 'incongruent_correct'
            Run2_Incongruent = [Run2_Incongruent; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        elseif strtrim(Run2_onsetTimes.trial_type(i,:)) == 'incongruent_incorrect'
            Run2_Incongruent = [Run2_Incongruent; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        elseif strtrim(Run2_onsetTimes.trial_type(i,:)) == 'congruent_correct'
            Run2_Congruent = [Run2_Congruent; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        elseif strtrim(Run2_onsetTimes.trial_type(i,:)) == 'congruent_incorrect'
            Run2_Congruent = [Run2_Congruent; Run2_onsetTimes.onset(i,:) Run2_onsetTimes.duration(i,:)];
        end
    end


    % Save timing files into text files

    save('incongruent_run1.txt', 'Run1_Incongruent', '-ASCII');
    save('incongruent_run2.txt', 'Run2_Incongruent', '-ASCII');
    save('congruent_run1.txt', 'Run1_Congruent', '-ASCII');
    save('congruent_run2.txt', 'Run2_Congruent', '-ASCII');

    % Go back to Flanker directory

    cd ../..

end

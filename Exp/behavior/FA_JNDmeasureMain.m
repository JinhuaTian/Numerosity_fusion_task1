% measure the JND
clear; close; 

%% -------------------------------------------------------------
%                       Set up Parameters
%----------------------------------------------------------------------
%% Screen
Distance = 60 ;
LengthScr = 35 ;
%window_area = [0 0 1200 900];
Screen('Preference', 'SkipSyncTests', 1);          % '1':  skip synchronism test;  '0': doesn't skip
window_area= [];

%% Experiment variebles
Nums  =  [8,10,13,16,20,25,32];
FAs  = [1,2,3,4,5,6,7];
PicIDs = 1:12;

Tasks = {'num','fa'};

%PicRptTimes = 1;
TriPerBlks =  [14,98]  ;
% SampTriPerBlks = [9,90] ;      % which should be mutiples of 9
% MatchTriPerBlks = [3,10];
%BlksPerSession =  1;          %  which should be mutiples of 2, and divisor of  PicPrtTimes
BlkNum = 8 ; 
taskType = 2; % MEG, fMRI
tasklist = fullfact([length(Tasks), BlkNum/ length(Tasks)/taskType,2]);
% generate task order. 1: num ; 2: fa    list2:the sequence  list3 fMRI/MEG tasks

while 1
    taskorder = tasklist(randperm(BlkNum),:);
    repeatLimit = 1;
    countRepeat = 0;
    countRepeatMax = 0;
    for k = 2:BlkNum
        if taskorder(k,3) == taskorder(k-1,3)
            countRepeat = countRepeat +1;
            if countRepeat > countRepeatMax
                countRepeatMax = countRepeat;
            end
        else
            countRepeat = 0;
        end
    end
    if countRepeatMax > repeatLimit
        continue;
    else
        break;
    end
end

ptaskorder = repmat([1,2],1,5);

%% Stimulus，fixation and time-relevant params
GrnFixDeg = 0.1;
RedFixDeg = 0.4;
PicDeg = 9;
InstruT  = 2;
stimT = 0.2;        % duration of each stimuli
% blankT = 0.6 ;
waitRspT = 4;
fixTmin =  0.5;      % min of jitter
fixTmax =  0.7;      % max of jitter
%RespT =  5;         % max of response time
InstrDeg = 30;
fixlDeg = 0.6;  % 1 degree.
fixwDeg = 0.15;  %0.05 degree

%% --------------------------------------------------------------
%                       Get subject Information
%%----------------------------------------------------------------------
prompt = {'SubjsNo:', 'Name(pinyin):', 'Session:','Stage(0:prac;1:main):'};  % description of fields

title = 'Subject information';
defaults = {'1', 'NaN','1','0'};
answer = inputdlg(prompt, title, 1.5, defaults);                         % opens dialog
SubID = answer{1, :};
Name = answer{2, :};
theBlk = str2double(answer{3, :});
stage = str2double(answer{4,:}) ; 
rootDir = pwd;

if stage == 0 
  TriPerBlk = TriPerBlks(1);
else 
  TriPerBlk = TriPerBlks(2);
end

%% calculate parameters
%BlkNum = length(Nums)* length(FAs)* length(Tasks) * PicRptTimes / TriPerBlk ;   % total number of blocks including judge size and judge number
%BlockNum =  BlkNum / BlksPerBlk;
%CondRptPerBlk = SampTriPerBlk / (length(Nums) * length(FAs)) ;
%FullOrder = fullfact([length(PicIDs),PicRptTimes]);


%% -------------------------------------------------------------
%                       Set up PTB enviorment
%----------------------------------------------------------------------

% define color
ScrNum = max(Screen('Screens'));
white = WhiteIndex(ScrNum);
black = BlackIndex(ScrNum);
grey  = 128;

% open background window and get the number of pixels

[windowPtr, wRect]=Screen('OpenWindow', ScrNum, grey, window_area); %the name of the opened window is windowPtr
[cX, cY] = RectCenter(wRect);
ifi = Screen('GetFlipInterval', windowPtr);  % second per frame, an estimate of the monitor flip interval

% calculate some param
PixelScr = wRect(3);
dpcm = PixelScr / LengthScr;
PixPerDeg = tan(deg2rad(1)) * Distance * dpcm;
deg2pix = @(x) round((x * PixPerDeg));
PicPix = deg2pix(PicDeg);
InstrPix = deg2pix(InstrDeg);
fixlpix = deg2pix(fixlDeg);                 %根据注视点长度视角，计算注视点的pixel个数
fixwpix = deg2pix(fixwDeg);

% Position of  instruc / stim / fixation
StimPst = CenterRect([0,0,PicPix,PicPix], wRect);
InstrPst = CenterRect([0,0,InstrPix,InstrPix], wRect);
fixrect = [0,0,fixlpix, fixwpix; 0,0,fixwpix, fixlpix];
fixPosn1 = CenterRect(fixrect(1,:), wRect);
fixPosn2 = CenterRect(fixrect(2,:), wRect);
fixPosn = [fixPosn1; fixPosn2]';
% Defining keys for response
KbName('UnifyKeyNames');
global confirmKey quitKey keyIndex
confirmKey = KbName('return');
quitKey = KbName('escape');
firstKey = KbName('LeftArrow');
secondKey = KbName('RightArrow');
keyIndex = [firstKey, secondKey];
HideCursor;


% tasklist = repmat( 1:2,1,BlkNum/2);   
% taskorder = Shuffle(tasklist);
% used for trial
%r= 1; b= 1;
%r= 10; b = 6;
% r = 25; b = 6;

%% -------------------------------------------------------------
%                       Main experiments
%----------------------------------------------------------------------
% used for trial
%r= 1; b= 1;
%r= 10; b = 6;
% r = 25; b = 6;
for r = theBlk: BlkNum
    if taskorder(r,3) == 1 % 
        taskName = 'fMRI';
        blankTmin = 3.5;
        blankTmax = 5.5;
    elseif taskorder(r,3) == 2
        taskName = 'MEG';
        blankTmin = 0.5; 
        blankTmax = 0.7;
    end
    %%  design matrix
    Matrix = table();
    if stage == 1  % main exp (0:prac)
        Task = taskorder(r,1);
        TaskSeq = taskorder(r,2);
        TIMES = TriPerBlk / (length(Nums) * length(FAs));
        picNos  = (PicIDs( (TaskSeq  - 1) * TIMES+ 1: TaskSeq * TIMES))';
        rpicNos = Shuffle(( TaskSeq -1 ) * TriPerBlk + 1:  TaskSeq * TriPerBlk);
        %Matrix.Blk = repelem([1:BlksPerBlk],TrialPerBlk)';
        Cond = fullfact([length(Nums), length(FAs),TriPerBlk/(length(Nums) * length(FAs))]);
    else
        % task 生成一个 1212 的序列
        Task = ptaskorder(r);
        % 随机一个 no（13-20) , 生成 6 * 6 ， 但shuffle后只取前12个
        picNos = randi([6, 10]);
        rpbegin = randi([200,230]);
        rpicNos = (rpbegin: rpbegin+ TriPerBlk-1);
        % ref:  随机一个 数（500-550）， 顺着排12个
        Cond = fullfact([length(Nums), length(FAs),1]);  
    end
    Matrix.TaskType = zeros(TriPerBlk,1)+ Task;                   %  '1': num ; '2':fa
    Scond = Cond(randperm(length(Cond)),:);
    CondOrder = Scond(1:TriPerBlk,:);              %
    Matrix.Num = Nums(CondOrder(:,1))';                      %
    Matrix.FA = FAs(CondOrder(:,2))';                       %
    Matrix.picNo = picNos(CondOrder(:,3));                 %
    Matrix.rpicNo =  rpicNos' ;                             %
    
    Matrix.Trial = (1: TriPerBlk)';                                    %    
    TriType = repmat([1,2], 1,TriPerBlk/2);
    TriOrder = Shuffle(TriType)';
    Matrix.TriType = TriOrder;              %
    
    %% show instruction
    instructArray = imread([rootDir,'\instructions\','Instruction.PNG']);
    imgPtr_instruct = Screen('MakeTexture',windowPtr, instructArray);
    Screen('FillRect', windowPtr, grey);
    Screen('DrawTextures', windowPtr, imgPtr_instruct, [],[]);
    Screen('DrawText',windowPtr,['Block: ', num2str(r)], cX-30,wRect(4)-300);
    Screen('DrawingFinished', windowPtr);
    Screen('Flip', windowPtr);
    WaitSecs(0.3);
    while KbCheck; end  % clear the keypress information in cache.
    while 1
        [~, ~, keyCode] = KbCheck;
        if keyCode(confirmKey)
            Screen('FillRect', windowPtr, grey);
            Screen('Flip', windowPtr);
            break
        elseif keyCode(quitKey)
            Priority(0); % resets priority
            sca;
            return
        end
        WaitSecs(0.05);
    end
    WaitSecs(0.3);
    
    %%  display the task instruction
    switch Task
        case 1              % NUM TASK
            instruction = 'num_instruction.PNG';
        case 2             % FA TASK
            instruction = 'fa_instruction.PNG';
    end
    instructArray = imread([pwd,'\instructions\',instruction]);
    imgPtr_instruct = Screen('MakeTexture',windowPtr, instructArray);
    Screen('FillRect', windowPtr, grey);
    Screen('DrawTextures', windowPtr, imgPtr_instruct, [],[]);
    Screen('DrawingFinished', windowPtr);
    Screen('Flip', windowPtr);
    WaitSecs(3);         % SHOW 2 SECONDS
    
    %%  Main loop 
        for t = 1: TriPerBlk
        
            % define the order & cond
            order = Matrix.TriType(t);
            num = Matrix.Num(t);
            fa = Matrix.FA(t);
            picNo = Matrix.picNo(t);
            rpicNo = Matrix.rpicNo(t);
            % read the imgs 
            TestImgAdd = sprintf('%s/stimulus/Num%d_FA%d/',rootDir,num,fa);
            TestImgName = sprintf('%d.png',picNo);
            TestImgArray = imread([TestImgAdd, TestImgName]);
            RefImg = sprintf('%s/stimulus/reference/%d.png',rootDir,rpicNo);
            RefImgArray = imread(RefImg);
            % blank
            Screen('FillRect',windowPtr,grey);
            WaitSecs(0.2);
            %fixation
            Screen('FillRect',windowPtr,grey);
            Screen('FillRect', windowPtr, black, fixPosn);
            vbl = Screen('Flip',windowPtr);
            fixT = unifrnd(fixTmin,fixTmax);%delay 

            %first sti
            if order == 1
                img1 = Screen('MakeTexture',windowPtr,TestImgArray);
            else 
                img1 = Screen('MakeTexture',windowPtr,RefImgArray);
            end
            Screen('DrawTexture', windowPtr,img1, [], StimPst);
            Screen('DrawingFinished', windowPtr);
            vbl = Screen('Flip',windowPtr,vbl + fixT -0.5*ifi);
            
            % blank
            Screen('FillRect',windowPtr,grey);
            vbl = Screen('Flip', windowPtr, vbl + stimT - 0.5*ifi); 
            
            % second sti
             if order == 1
                img2 = Screen('MakeTexture',windowPtr,RefImgArray);
            else 
                img2 = Screen('MakeTexture',windowPtr,TestImgArray);
            end
            Screen('DrawTexture', windowPtr,img2, [], StimPst);
            Screen('DrawingFinished', windowPtr);
            blankT = unifrnd(blankTmin,blankTmax); % randmonize isi
            vbl = Screen('Flip',windowPtr,vbl + blankT -0.5*ifi);
            t0 = vbl;
            % get response
            [keyPressed,Response,T] = GetResp(stimT,vbl,t0);
            
            if keyPressed == 0
                % blank & get response
                Screen('FillRect',windowPtr,grey);
                vbl = Screen('Flip', windowPtr, vbl + stimT - 0.5*ifi); 
                [~,Response,T] = GetResp(waitRspT,vbl,t0);
            end
            Matrix.Rep(t) = Response;
            
            Matrix.RT(t) = T;
            % Judge
            if Task == 1    % num
                comp = (num < 16);
                
            else
                comp =( fa < 4);
            end
            if  Response ~= 9  && ((comp==0 && Response == order ) || (comp==1 && Response ~= order))
                Judge =  1; %right
            else
                Judge = 2; % false
            end
            Matrix.Judge(t) = Judge ;
        end
        
     % calculate and show 
     Nerror = length(find(Matrix.Judge == 2));
     Screen('DrawText',windowPtr,['Error:' num2str(Nerror)],cX-50,cY-200);
     Screen('Flip',windowPtr);
     WaitSecs(2.0);
     KbWait;
     
     %  save the matrix
     outputDir = fullfile(pwd,'fa_rawdata');
     if ~exist(outputDir, 'dir')
         mkdir(outputDir);
     end
     c = clock;
     
    if stage == 0
        foretitle = 'Prac_';
    else 
        foretitle  = '';
    end
     outputFileName = [foretitle 'S0' SubID '_' Name '_' taskName '_' Tasks{Task} '_Run' num2str(r) '_' num2str(c(2)) '_' num2str(c(3)) '_' ,...
        num2str(c(4)) '_' num2str(c(5)) '.mat'];
     
    Data.SubID = SubID;
    Data.Name = Name;
    Data.ExpTime = c;
    Data.Nerror = Nerror;
    Data.Block = theBlk;
    
    Data.Main = Matrix;
    Data.Task = Tasks{Task};
    save([outputDir,'/',outputFileName], 'Data');
end

ShowCursor();
Priority(0);
sca;

function [keyPressed,Response,RT] = GetResp(t,vbl,t0)
global quitKey keyIndex
while KbCheck; end  % clear the keypress information in cache.
keyPressed = 0;
Response = 99;
RT = -99;
while GetSecs- vbl < t  && ~keyPressed
    [~, secs, keyCode] = KbCheck;
    if keyCode(quitKey)
        Priority(0); %resets priority
        sca;
        return
    elseif any(keyCode(keyIndex))
        keyPressed = 1;
        Response = find(keyCode(keyIndex)==1,1);
        while KbCheck; end
        RT = secs - t0;
    end
end
end
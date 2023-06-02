
%  exp 03
clear;


Screen('Preference', 'SkipSyncTests', 0);          % '1':  skip synchronism test;  '0': doesn't skip
%window_area = [0 0 600 400];
window_area= [];

%global marker address
%marker = 1;             % 0: test on normal computer     1:send marker

%if marker == 1
    port_id = 'D100';
    ioObj = io64;
    status = io64(ioObj); %#ok<*NASGU>
    address = hex2dec(port_id);
%end
%% -------------------------------------------------------------
%                       Set up Parameters
%----------------------------------------------------------------------
%% Experiment variebles

Nums  =  [8, 16, 32];
FAs  = [1, 2, 3];
PicIDs = (1:5);
Tasks = {'num','fa'};
PicRptTimes = 28;
SampTriPerBlks = [9,90] ;      % which should be mutiples of 9
MatchTriPerBlks = [3,10];

BlksPerRun =  2;          %  which should be mutiples of 2, and divisor of  PicPrtTimes


jndcs = [];
%% Screen
Distance = 60 ;
LengthScr = 40 ;

%% Stimulus，fixation and time-relevant params
GrnFixDeg = 0.1;
RedFixDeg = 0.3;
PicDeg = 9;
InstruT  = 2;
stimT = 0.2;        % duration of each stimuli
blankT = 0.6 ;
ISImin =  0.5;      % min of jitter
ISImax =  0.7;      % max of jitter
%RespT =  5;         % max of response time
InstrDeg = 30;

%% --------------------------------------------------------------
%                       Get subject Information
%%----------------------------------------------------------------------
prompt = {'被试编号:', '姓名（请填写拼音，例：Xiaoming):', 'Session:','阶段(0:prac;1:mian)：'};  % description of fields
title = 'Subject information';
defaults = {'1', 'NaN','1','0'};
answer = inputdlg(prompt, title, 1.3, defaults);                         % opens dialog
SubID = answer{1, :};
Name = answer{2, :};
theRun = str2double(answer{3, :});
stage = str2double(answer{4,:});
rootDir = pwd;


if stage == 0 
   SampTriPerBlk = SampTriPerBlks(1);
   MatchTriPerBlk = MatchTriPerBlks(1);
else 
   SampTriPerBlk = SampTriPerBlks(2);
   MatchTriPerBlk = MatchTriPerBlks(2);
end

TrialPerBlk = SampTriPerBlk + MatchTriPerBlk;
BlkNum = length(Nums)* length(FAs)* length(Tasks) * length(PicIDs)* PicRptTimes / SampTriPerBlk ;   % total number of blocks including judge size and judge number
RunNum =  BlkNum / BlksPerRun;
CondRptPerBlk = SampTriPerBlk / (length(Nums) * length(FAs)) ;
FullOrder = fullfact([length(PicIDs),PicRptTimes]);


%% -------------------------------------------------------------
%                       Set up PTB enviorment
%----------------------------------------------------------------------

Screen('Preference', 'SkipSyncTests', 1);   
% define color
ScrNum = max(Screen('Screens'));
white = WhiteIndex(ScrNum);
black = BlackIndex(ScrNum);
grey  = 128;
red   = [255 0 0];
green = [0 255 0];

% open background window and get the number of pixels

[windowPtr, wRect]=Screen('OpenWindow', ScrNum, grey, window_area); %the name of the opened window is windowPtr
[cX, cY] = RectCenter(wRect);
ifi = Screen('GetFlipInterval', windowPtr);  % second per frame, an estimate of the monitor flip interval

% calculate some param
PixelScr = wRect(3);
dpcm = PixelScr / LengthScr;
PixPerDeg = tan(deg2rad(1)) * Distance * dpcm;
deg2pix = @(x) round(x * PixPerDeg);
GrnFixPix = deg2pix(GrnFixDeg);
RedFixPix = deg2pix(RedFixDeg);
PicPix = deg2pix(PicDeg);
InstrPix = deg2pix(InstrDeg);

% Position of  instruc / stim
StimPst = CenterRect([0,0,PicPix,PicPix], wRect);
InstrPst = CenterRect([0,0,InstrPix,InstrPix], wRect);

% Defining keys for response
KbName('UnifyKeyNames');
confirmKey = KbName('return');
quitKey = KbName('escape');
firstKey = KbName('LeftArrow');
secondKey = KbName('RightArrow');
keyIndex = [firstKey, secondKey];
HideCursor;

% used for trial
%r= 1; b= 1;
%r= 10; b = 6;
% r = 25; b = 6;

for r = theRun: RunNum
    
    %  design matrix
    Matrix = table();
    Matrix.Blk = repelem((1:BlksPerRun),TrialPerBlk)';
    
    %     for k = 1:BlksPerRun
    %         i1 = (k-1) * TrialPerBlk +1;
    %         i2 = (k-1) * TrialPerBlk +TrialPerBlk ;
    %         Matrix.Blk([i1:i2]) = k ;
    %     end
    
    Matrix.Trial = repmat((1: TrialPerBlk)',[BlksPerRun,1]);
    Matrix.TrialType = zeros(height(Matrix),1);                   %  '0': sample trial ; '1': match trial
    t = randi([2, TrialPerBlk/MatchTriPerBlk ], MatchTriPerBlk * BlksPerRun ,1);
    for i = 1: length(t)
        t(i) = (i-1)* TrialPerBlk/MatchTriPerBlk + t(i);
    end
    Matrix.TrialType(t) = 1;
    
    %  shuffle the num & fa picID
    Rindex = (r-1) * BlksPerRun * 0.5  * CondRptPerBlk +1;
    RunOrder = FullOrder (Rindex: Rindex + BlksPerRun * CondRptPerBlk/2 -1 ,1);
    NumOrder = randperm(BlksPerRun/ 2);
    FAOrder = randperm (BlksPerRun/ 2);
    
    % random  the first tasks
    Task1 = randi([1,2]);
    TaskOrder = table();   %TaskOrder = zeros(BlksPerRun,2);
    TaskOrder.Blk =  (1:BlksPerRun)';
    TaskOrder.task = repmat([Task1, 3-Task1], [1,BlksPerRun/2])';
    
    % list all the blk cond and PicNo
    n = 1; fa = 1;
    for p = 1: height(TaskOrder)
        if TaskOrder.task(p) ==  1
            TaskOrder.withincon(p) = NumOrder(n);
            n = n+1;
        else
            TaskOrder.withincon(p)  = FAOrder(fa);
            fa = fa+1;
        end
        TaskOrder.PicNo{p} = RunOrder(TaskOrder.withincon(p) : (TaskOrder.withincon(p)+CondRptPerBlk-1))';
    end
    
    %  instruction
    
    %[instructArray,~,alpha] = imread([rootDir,'/', 'Instruction.PNG']);
    %instructArray(:,:,4) = alpha(:,:);
    instructArray = imread([rootDir,'\instructions\','Instruction.PNG']);
    imgPtr_instruct = Screen('MakeTexture',windowPtr, instructArray);
    Screen('FillRect', windowPtr, grey);
    Screen('DrawTextures', windowPtr, imgPtr_instruct, [],[]);
    Screen('DrawText',windowPtr,['Session: ', num2str(r)], cX-30,wRect(4)-300);
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
    
    
    %% block loops
    for b = 1: BlksPerRun
        Task =  TaskOrder.task(b);
  %      PicNo = cell2mat(TaskOrder.PicNo(b));
        
 %       order = ((r-1) * BlksPerRun + b-1) * CondRptPerBlk +1;
%        cond = FullOrder(order: order+CondRptPerBlk-1 , :);
        
        
        Conds = fullfact([length(Nums), length(FAs),5]);
        %Conds(:,4) =  PicNo(Conds(:,3)) ;
        Conds(:,1) = Nums(Conds(:,1)) ;
        n1 = randperm(45);
        n2 = randperm(45);
        Condition = [Conds(n1,:);Conds(n2,:)];
        
        
       % t = randperm(SampTriPerBlk);
       % Condition = Conds(t, [1,2,4]);
        
        % show instruction
        %q = GetSecs();
        switch Task
            case 1              % NUM TASK
                instruction = 'num_instruction.PNG';
                %MakeMark(71,q);
                io64(ioObj, address, 71);
                WaitSecs(0.01);
                io64(ioObj,address,0);
                
            case 2             % FA TASK
                instruction = 'fa_instruction.PNG';
                %MakeMark(72,q);
                %MakeMark(49,vbl);
                io64(ioObj, address, 72);
                WaitSecs(0.01);
                io64(ioObj,address,0);

        end
        
        
        % [instructArray,~,alpha] = imread([rootDir,'/', 'instruction_formal.png']);
        % instructArray(:,:,4) = alpha(:,:);
        instructArray = imread([pwd,'\instructions\',instruction]);
        imgPtr_instruct = Screen('MakeTexture',windowPtr, instructArray);
        Screen('FillRect', windowPtr, grey);
        Screen('DrawTextures', windowPtr, imgPtr_instruct, [],[]);
        Screen('DrawingFinished', windowPtr);
        vbl = Screen('Flip', windowPtr);
        WaitSecs(2);         % SHOW 2 SECONDS
        
        % main exp
        sampleIndex = 1;
        for t = 1: TrialPerBlk
            TriIndex =  (b-1) * TrialPerBlk + t;
            TrialType = Matrix.TrialType(TriIndex);
            
            % fix & stim
            switch TrialType
                case  0                % sample trial
                    % green fixation
                    Screen('DrawDots',windowPtr,[0;0],GrnFixPix,green,[cX,cY],2);
                    %vbl = Screen('Flip',windowPtr, vbl + blankT  - 0.5 * ifi );
                    vbl = Screen('Flip',windowPtr);
                    % stim
                    num =  Condition(sampleIndex,1);
                    fa = FAs(Condition(sampleIndex,2));
                    picNo = Condition(sampleIndex,3);
                    
                    ImageAdd = sprintf('%s/stimulus/Num%d_FA%d/',pwd,num,fa);
                    ImageName  = sprintf('%d.png',picNo);
                    %    [ImageArray,~,alpha] = imread([ImageAdd,ImageName]);
                    %   ImageArray(:,:,4) = alpha(:,:);
                    ImageArray = imread([ImageAdd,ImageName]);
                    
                    img = Screen('MakeTexture',windowPtr,ImageArray);
                    Screen('DrawTexture', windowPtr,img, [], StimPst);
                    Screen('DrawDots',windowPtr,[0;0],GrnFixPix,green,[cX,cY],2);
                    Screen('DrawingFinished', windowPtr);
                    fixT = unifrnd(ISImin,ISImax); %delay
                    %  vbl = Screen('Flip',windowPtr);
                    vbl = Screen('Flip',windowPtr,vbl + fixT -0.5*ifi);
                    
                    % Pic Trigger
                    n = find(Nums == num);  i = length(FAs);    p = length(PicIDs);
                    ID =  (n-1)*i*p + (fa -1)*p +  picNo;

                    io64(ioObj, address, ID);
                     while GetSecs - vbl < 0.01
                     end
                     io64(ioObj,address,0);

                   
                    t0 = vbl;
                    
                    % escape chance
                    Response = -1;
                    RT = -1;
                    while KbCheck; end  % clear the keypress information in cache.
                    keyPressed = 0;
                    while GetSecs- vbl <= stimT  %&& ~keyPressed
                        [keyIsDown, secs, keyCode] = KbCheck;
                        if keyCode(quitKey)
                            Priority(0); %resets priority
                            sca;
                            return
                        elseif any(keyCode(keyIndex))
                            keyPressed = 1;
                            Response = find(keyCode(keyIndex)==1,1);
                             while KbCheck; end 
                             if Response == 1
                                %MakeMark(49,vbl);
                                io64(ioObj, address, 49);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);                                
                            elseif Response  == 2
                                %MakeMark(51,vbl);
                                io64(ioObj, address, 51);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);
                            end
%                             if Response == 1
%                                 MakeMark(49,vbl);
%                             elseif Response  == 2
%                                 MakeMark(51,vbl);
%                             end
                     
                            RT = secs - t0;
                        end
                    end
                    
                    Screen('FillRect',windowPtr,grey);
                    vbl = Screen('Flip', windowPtr, vbl + stimT - 0.5*ifi);                   
                    while KbCheck; end  % clear the keypress information in cache.
                    keyPressed = 0;
                    while GetSecs- vbl <= blankT % && ~keyPressed
                        [keyIsDown, secs, keyCode] = KbCheck;
                        if keyCode(quitKey)
                            Priority(0); %resets priority
                            sca;
                            return
                        elseif any(keyCode(keyIndex))
                            keyPressed = 1;
                            Response = find(keyCode(keyIndex)==1,1);
                             while KbCheck; end
                             
                             if Response == 1
                                %MakeMark(49,vbl);
                                io64(ioObj, address, 49);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);                                
                            elseif Response  == 2
                                %MakeMark(51,vbl);
                                io64(ioObj, address, 51);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);
                            end
%                             if Response == 1        % leftarrow
%                                 MakeMark(49,vbl);
%                             elseif Response  == 2   % rightarrow
%                                 MakeMark(51,vbl);
%                             end  
                            RT = secs - t0;
                        end
                    end
                    
                    %                     while KbCheck; end  % clear the keypress information in cache.
                    %                     keyPressed = 0;
                    %                     while GetSecs- vbl <=  stimT && ~keyPressed
                    %                         [keyIsDown, secs, keyCode] = KbCheck;
                    %                         if keyCode(quitKey)
                    %                             Priority(0); %resets priority
                    %                             sca;
                    %                             return
                    %                         end
                    %                     end
                    %
                    %                     Screen('FillRect',windowPtr,grey);
                    %                     vbl = Screen('Flip', windowPtr, vbl + stimT - 0.5*ifi);
                    %
                    %                     keyPressed = 0;
                    %                     while GetSecs- vbl <= blankT  && ~keyPressed
                    %                         [keyIsDown, secs, keyCode] = KbCheck;
                    %                         if keyCode(quitKey)
                    %                             Priority(0); %resets priority
                    %                             sca;
                    %                             return
                    %                         end
                    %                     end
                    
                    sampleIndex = sampleIndex + 1;
                    
                    Judge = -1;
                    jndc = 0;
                    
                case  1               % match trial
                    
                    ID =99;
                    % red  fiaxtion
                    Screen('DrawDots',windowPtr,[0;0],RedFixPix,red,[cX,cY],2);
                    vbl = Screen('Flip',windowPtr );
                    
                    % stim  & fix
%                     tnums = Nums;
%                     tnums(Nums == Condition(sampleIndex-1,1)) = [];
%                     num = tnums(randi(1:length(tnums)));
%                     tfas = FAs;
%                     tfas(FAs == Condition(sampleIndex-1,2)) = [];
%                     %
%                     %                     tnums(find(Nums == Condition(t-1,1))) = [];
%                     %                     num = tnums(randi(1:length(tnums)));
%                     %                     tias = IAs;
%                     %                     tias(find(IAs == Condition(t-1,2))) = [];
%                     fa = tfas(randi(1:length(tfas)));
                    
                    num = Condition(sampleIndex-1,1);
                    fa = Condition(sampleIndex-1,2);
                    jndc = randi([1,2]);
                    %jndcs = [jndcs;jndc];
                    
                    picNo = randi([1,5]);
                    
                    switch Task
                        case 1              % NUM TASK
                            ImageAdd = sprintf('%s/m_num_stimulus/%dNum%d_FA%d/',pwd,jndc,num,fa);
                            
                        case 2             % FA TASK
                            ImageAdd = sprintf('%s/m_fa_stimulus/%dNum%d_FA%d/',pwd,jndc,num,fa);
                    end

                    
                    
                    ImageName  = sprintf('%d.png',picNo);
                    ImageArray = imread([ImageAdd, ImageName]);
                    img = Screen('MakeTexture',windowPtr,ImageArray);
                    Screen('DrawTexture', windowPtr,img, [], StimPst);
                    Screen('DrawDots',windowPtr,[0;0],RedFixPix,red,[cX,cY],2);
                    Screen('DrawingFinished', windowPtr);
                    fixT = unifrnd(ISImin,ISImax); %delay
                    vbl = Screen('Flip',windowPtr,vbl + fixT -0.5*ifi);
                    t0 = vbl;
                    
                    % Match  trigger
%                     MakeMark(99,vbl);
%                     
                     io64(ioObj, address, 99);
                     while GetSecs - vbl < 0.01
                     end
                     io64(ioObj,address,0);
                                 
                    % get response
                    while KbCheck; end  % clear the keypress information in cache.
                    keyPressed = 0;
                    while GetSecs- vbl <= stimT  %&& ~keyPressed
                        [keyIsDown, secs, keyCode] = KbCheck;
                        if keyCode(quitKey)
                            Priority(0); %resets priority
                            sca;
                            return
                        elseif any(keyCode(keyIndex))
                            keyPressed = 1;
                            Response = find(keyCode(keyIndex)==1,1);
                             while KbCheck; end 
                            if Response == 1
                                %MakeMark(49,vbl);
                                io64(ioObj, address, 49);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);                                
                            elseif Response  == 2
                                %MakeMark(51,vbl);
                                io64(ioObj, address, 51);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);
                            end
                          
                            RT = secs - t0;
                        end
                    end
                    
                    Screen('FillRect',windowPtr,grey);
                    vbl = Screen('Flip', windowPtr, vbl + stimT - 0.5*ifi);
                    
                    while KbCheck; end  % clear the keypress information in cache.
                    keyPressed = 0;
                    while GetSecs- vbl <= 1  %&& ~keyPressed
                        [keyIsDown, secs, keyCode] = KbCheck;
                        if keyCode(quitKey)
                            Priority(0); %resets priority
                            sca;
                            return
                        elseif any(keyCode(keyIndex))
                            keyPressed = 1;
                            Response = find(keyCode(keyIndex)==1,1);
                             while KbCheck; end 
                             if Response == 1
                                 %MakeMark(49,vbl);
                                 %ioObj = io64;
                                 io64(ioObj, address, 49);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);
                             elseif Response  == 2
                                 %MakeMark(51,vbl);
                                 %ioObj = io64;
                                 io64(ioObj, address, 51);
                                 while GetSecs - vbl < 0.01
                                 end
                                 io64(ioObj,address,0);
                             end
                            RT = secs - t0;
                        end
                    end
                    
                    % Judge
                           
%                     if Task ==  1            % numtask
%                         if  (num > Condition(sampleIndex-1,1) && Response ==2) || (num < Condition(sampleIndex-1,1) && Response ==1)
%                             Judge = 1;
%                         else
%                             Judge = 2;
%                         end
%                     else   % fa task
%                         if  (fa > Condition(sampleIndex-1,2) && Response ==2) || (fa < Condition(sampleIndex-1,2) && Response ==1)
%                             Judge = 1;
%                         else
%                             Judge = 2;
%                         end
%                     end
%                     
                    
                    if jndc == Response
                        Judge =1;
                    else
                        Judge =2;
                    end
            end
            
            % blank
            
            %              Screen('FillRect',windowPtr,grey);
            %              vbl = Screen('Flip', windowPtr, vbl + stimT - 0.5*ifi);
            
            %           WaitSecs(0.1);
            
            Matrix.Num(TriIndex) = num ;
            Matrix.FA(TriIndex) = fa;
            Matrix.PicNo(TriIndex) = picNo;
            Matrix.picID(TriIndex) = ID;
            Matrix.Rep(TriIndex) = Response ;
            Matrix.RT(TriIndex) = RT   ;
            Matrix.Judge(TriIndex) = Judge;
            Matrix.BlkTask(TriIndex) = Task;  % 1; num ; 2: fa
            Matrix.JNDcon(TriIndex) = jndc;
            
        end
               
    end
    
    %  Judge
    Merror = sum((Matrix.Judge(Matrix.Judge == 2)))/2;
    %                 nanmean(abs(Matrix.Judge));
    %                 NMiss = sum(isnan(Matrix.Judge));
    Screen('DrawText',windowPtr,['Error: ' num2str(Merror)], cX-50,cY-200);
    %                Screen('DrawText',windowPtr,['Miss: ' num2str(NMiss)], cX-50,cY-100);
    Screen('Flip', windowPtr);
    WaitSecs(2.0);
    KbWait;
    
    
    %% save the data
    
    
    outputDir = fullfile(rootDir, 'data');
    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    c = clock;
    
    if stage == 0
        foretitle = 'Prac';
    else 
        foretitle  = '';
    end
    outputFileName = [foretitle 'S0' SubID '_' Name '_Run' num2str(r) '_' num2str(c(1)) '_' num2str(c(2)) '_' num2str(c(3)) '_' ,...
        num2str(c(4)) '_' num2str(c(5)) '.mat'];
    
    Data.SubID = SubID;
    Data.Name = Name;
    Data.ExpTime = c;
    Data.Block = theRun;
    Data.RawMain = Matrix;
    save([outputDir,'/',outputFileName], 'Data');
    
    
end


%% ------------------------------------------------------------
%                 restore intial settings
%----------------------------------------------------------------------
ShowCursor();  % shows the cursor
Priority(0);  % resets priority
sca;


%  make marker 
% function MakeMark(n,vbl)
% global marker address
% if marker == 1
%     ioObj = io64;
%     io64(ioObj, address, n);
%     while GetSecs - vbl < 0.01
%     end
%     io64(ioObj,address,0);
% end
% end


%function responsedetect 
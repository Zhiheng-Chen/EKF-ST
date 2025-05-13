clear;
clc;
close all;

rng(0);

%% simulation
%system parameters
params_sys = struct();
params_sys.sigma = 10;
params_sys.beta = 8/3;
params_sys.rho = 28;

%process noise
var_w = 0.1^2;
var_w_theta = 1e-4;
Q = blkdiag(var_w.*eye(3),var_w_theta.*eye(4));

%simulate with process noise
X0 = [1 1 1]';
Deltat = 0.001;
t_out = 0:Deltat:120;
X_out = zeros(3,length(t_out));
arr_w = sqrt(var_w).*randn(3,length(t_out));
X_out(:,1) = X0;

for ii = 1:length(t_out)-1
    w = arr_w(:,ii);
    [~,X] = ode45(@(t,X) lorenzEqs(t,X,params_sys,w),[t_out(ii),t_out(ii+1)],X_out(:,ii));
    X_out(:,ii+1) = X(end,:)';
end

%measurements
var_v = 1^2;
R = var_v*eye(3);
H = [eye(3),zeros(3,4)];
arr_v = sqrt(var_v).*randn(3,length(t_out));
z = X_out(1:3,:)+arr_v;

%% EKF-ST
%initializations
x_k_k = [X0;10.*ones(4,1)];
P_k_k = diag([10,10,10,30,30,30,30]);
arr_x_u = zeros(7,length(t_out));
arr_P_u = zeros(7,7,length(t_out));
arr_x_u(:,1) = x_k_k;
arr_P_u(:,:,1) = P_k_k;
lambda_ST = 0.1;    %thresold knob
arr_F = zeros(7,7,0);   %log Jacobian matrices for observability computations

%EKF-ST iterations (prediction-update-threshold loops)
for ii = 2:length(t_out)
    %prediction
    x_kp1_k = EKF_statePrediction(x_k_k,Deltat,params_sys);
    F_k = calcF(x_k_k,Deltat,params_sys);
    P_kp1_k = F_k*P_k_k*F_k'+Q;
    %Kalman gain
    K_kp1 = P_kp1_k*H'*inv(H*P_kp1_k*H'+R);
    %update
    x_kp1_kp1 = x_kp1_k+K_kp1*(z(:,ii)-H*x_kp1_k);
    P_kp1_kp1 = (eye(7)-K_kp1*H)*P_kp1_k*(eye(7)-K_kp1*H)'+K_kp1*R*K_kp1';
    %threshold
    smallInds = [false;false;false;abs(x_kp1_kp1(4:end))<lambda_ST];
    smallInds = find(smallInds);
    for k = smallInds
        x_kp1_kp1(k) = 0;
        P_kp1_kp1(:,k) = 0;
        P_kp1_kp1(k,:) = 0;
        P_kp1_kp1(k,k) = 0.5;
    end
    %log data
    arr_x_u(:,ii) = x_kp1_kp1;
    arr_P_u(:,:,ii) = P_kp1_kp1;
    x_k_k = x_kp1_kp1;
    P_k_k = P_kp1_kp1;
    arr_F(:,:,ii) = F_k;
end

%% show results
%plot true and estimated trajectories
figure;
plot3(X_out(1,:),X_out(2,:),X_out(3,:),'color',[0,0,1]);
hold on;
plot3(arr_x_u(1,:),arr_x_u(2,:),arr_x_u(3,:),'color',[1,0,0]);
grid on;
box on;
legend('True Trajectory','EKF-ST Estimate','location','northeast');

%plot state estimation error
%-error and bounds
e_x = X_out(1,:)-arr_x_u(1,:);
e_y = X_out(2,:)-arr_x_u(2,:);
e_z = X_out(3,:)-arr_x_u(3,:);
sigma_x = sqrt(reshape(arr_P_u(1,1,:),1,[]));
sigma_y = sqrt(reshape(arr_P_u(2,2,:),1,[]));
sigma_z = sqrt(reshape(arr_P_u(3,3,:),1,[]));
%-plot
figure;
%--e1
subplot(3,1,1);
fill([t_out,fliplr(t_out)],[2*sigma_x,fliplr(-2*sigma_x)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_x,'color',[1,0,0],'lineWidth',0.7);
xlim([0,30]);
ylim([-2,2]);
ylabel('e_x');
title('State Estimation Error and 2\sigma Bounds');
%--e2
subplot(3,1,2);
fill([t_out,fliplr(t_out)],[2*sigma_y,fliplr(-2*sigma_y)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_y,'color',[1,0,0],'lineWidth',0.7);
xlim([0,40]);
ylim([-2,2]);
ylabel('e_y');
%--e3
subplot(3,1,3);
fill([t_out,fliplr(t_out)],[2*sigma_z,fliplr(-2*sigma_z)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_z,'color',[1,0,0],'lineWidth',0.7);
xlim([0,40]);
ylim([-2,2]);
xlabel('Time (s)')
ylabel('e_z');

%plot coefficient estimation error
%-error and bound
e_theta1 = 0-arr_x_u(4,:);
e_theta2 = -params_sys.sigma-arr_x_u(5,:);
e_theta3 = params_sys.sigma-arr_x_u(6,:);
e_theta4 = 0-arr_x_u(7,:);
sigma_theta1 = sqrt(reshape(arr_P_u(4,4,:),1,[]));
sigma_theta2 = sqrt(reshape(arr_P_u(5,5,:),1,[]));
sigma_theta3 = sqrt(reshape(arr_P_u(6,6,:),1,[]));
sigma_theta4 = sqrt(reshape(arr_P_u(7,7,:),1,[]));
%-plot
figure;
%--e_theta1
subplot(4,1,1);
fill([t_out,fliplr(t_out)],[2*sigma_theta1,fliplr(-2*sigma_theta1)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_theta1,'color',[1,0,0],'lineWidth',0.7);
xlim([0,40]);
ylabel('e_\theta_1');
title('Basis Coefficient Estimation Error and 2\sigma Bounds');
%--e_theta2
subplot(4,1,2);
fill([t_out,fliplr(t_out)],[2*sigma_theta2,fliplr(-2*sigma_theta2)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_theta2,'color',[1,0,0],'lineWidth',0.7);
xlim([0,40]);
ylabel('e_\theta_2');
%--e_theta3
subplot(4,1,3);
fill([t_out,fliplr(t_out)],[2*sigma_theta3,fliplr(-2*sigma_theta3)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_theta3,'color',[1,0,0],'lineWidth',0.7);
xlim([0,40]);
ylabel('e_\theta_3');
%--e_theta4
subplot(4,1,4);
fill([t_out,fliplr(t_out)],[2*sigma_theta4,fliplr(-2*sigma_theta4)],[0,0,1],'faceAlpha',0.5,'edgeColor','none');  %2-sigma bound
hold on;
plot(t_out,e_theta4,'color',[1,0,0],'lineWidth',0.7);
xlim([0,40]);
ylabel('e_\theta_4');

%final parameter estimate
disp(arr_x_u(4:end,end));

%% functions
function X_dot = lorenzEqs(t,X,params_sys,w)
    sigma = params_sys.sigma;
    beta = params_sys.beta;
    rho = params_sys.rho;
    x = X(1);
    y = X(2);
    z = X(3);
    x_dot = sigma*(y-x)+w(1);
    y_dot = x*(rho-z)-y+w(2);
    z_dot = x*y-beta*z+w(3);
    X_dot = zeros(3,1);
    X_dot(1) = x_dot;
    X_dot(2) = y_dot;
    X_dot(3) = z_dot;
end

function x_kp1_k = EKF_statePrediction(x_k_k,Deltat,params_sys)
    sigma = params_sys.sigma;
    beta = params_sys.beta;
    rho = params_sys.rho;
    x = x_k_k(1);
    y = x_k_k(2);
    z = x_k_k(3);
    theta1 = x_k_k(4);
    theta2 = x_k_k(5);
    theta3 = x_k_k(6);
    theta4 = x_k_k(7);

    x_kp1_k = zeros(length(x_k_k),1);
    x_kp1_k(1) = x+(theta1+theta2*x+theta3*y+theta4*z)*Deltat;
    x_kp1_k(2) = y+(x*(rho-z)-y)*Deltat;
    x_kp1_k(3) = z+(x*y-beta*z)*Deltat;
    x_kp1_k(4:end) = x_k_k(4:end);    
end

function F = calcF(X,Deltat,params_sys)
    x = X(1);
    y = X(2);
    z = X(3);
    theta1 = X(4);
    theta2 = X(5);
    theta3 = X(6);
    theta4 = X(7);
    sigma = params_sys.sigma;
    rho = params_sys.rho;
    beta = params_sys.beta;
    
    F = eye(length(X));
    F(1,1) = 1+theta2*Deltat;
    F(1,2) = theta3*Deltat;
    F(1,3) = theta4*Deltat;
    F(2,1) = (rho-z)*Deltat;
    F(2,2) = 1-Deltat;
    F(2,3) = -x*Deltat;
    F(3,1) = y*Deltat;
    F(3,2) = x*Deltat;
    F(3,3) = 1-beta*Deltat;

    F(1,4) = Deltat;
    F(1,5) = Deltat*x;
    F(1,6) = Deltat*y;
    F(1,7) = Deltat*z;

    % F = [1+theta2*Deltat    theta3*Deltat   theta4*Deltat   Deltat  Deltat*x    Deltat*y    Deltat*z
    %      (rho-z)*Deltat     1-Deltat        -x*Deltat       0       0           0           0
    %      y*Deltat           x*Deltat        1-beta*Deltat   0       0           0           0
    %      0                  0               0               1       0           0           0
    %      0                  0               0               0       1           0           0
    %      0                  0               0               0       0           1           0
    %      0                  0               0               0       0           0           1];
end

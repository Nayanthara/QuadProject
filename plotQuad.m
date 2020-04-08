function plotQuad(x,y,theta,tL,tR,Lx,Ly)

persistent f

w = Lx/20;
h = Ly/30;

if isempty(f) || ~isvalid(f)
    f = figure(...
        'Toolbar','none',...
        'NumberTitle','off',...
        'Name',getString(message('rl:env:VizNameFlyingRobot')),...
        'Visible','on',...
        'MenuBar','none');
    
    ha = gca(f);
    localResetAxes(ha,Lx,Ly)
    grid(ha,'on');
    
    omega = 0.25;
    xMesh = linspace(0,100,100/0.4);
    yMesh = linspace(0,40,40/0.4);
    [X,Y] = meshgrid(xMesh,yMesh);
    windVel = -cos(omega*X).*cos(omega*Y);
    colormap(gray)
    contourf(ha,X,Y,windVel,100,'LineColor','none')
    hold on
    
    xMesh = linspace(0,100,101);
    yMesh = linspace(0,40,41);
    [X,Y] = meshgrid(xMesh,yMesh);
    U = -0.5*cos(omega*X).*sin(omega*Y);
    V = 0.5*sin(omega*X).*cos(omega*Y);
    quiver(ha,X,Y,U,V,'m')
    
end

ha = gca(f);

c = cos(theta);
s = sin(theta);
R = [c -s;s c];
T = [R [x y]';zeros(1,3)];

V0 = [  -w -w  w  w w 0      0    ;
    h -h -h  h 0 h*1.5 -h*1.5;
    ones(1,7)   ];
V1 = T*V0;

vx = V1(1,1:4);
vy = V1(2,1:4);
ux = [x V1(1,5)];
uy = [y V1(2,5)];
wx = V1(1,6:7);
wy = V1(2,6:7);
tx = c*[tL tR];
ty = s*[tL tR];

body = findobj(ha,'Tag','body');
nose = findobj(ha,'Tag','nose');
quiv = findobj(ha,'Tag','quiv');

if isempty(quiv)
    quiver(ha,wx,wy,tx,ty,'Color','r','LineWidth',2,'Tag','quiv');
else
    quiv.XData = wx;
    quiv.YData = wy;
    quiv.UData = tx;
    quiv.VData = ty;
end
if isempty(body)
    patch(vx,vy,'y','Tag','body');
else
    body.XData = vx;
    body.YData = vy;
end
if isempty(nose)
    line(ux,uy,'Color','k','LineWidth',1,'Tag','nose');
else
    nose.XData = ux;
    nose.YData = uy;
end

drawnow();

function localResetAxes(ha,Lx,Ly)
cla(ha);
set(ha,'XLim',[-Lx,Lx]);
set(ha,'YLim',[-Ly,Ly]);


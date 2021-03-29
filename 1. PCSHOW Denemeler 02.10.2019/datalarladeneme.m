A=[x(:),y(:),z(:)];
figure();
pcshow(A);
title('Sphere with Color Texture');
xlabel('X');
ylabel('Y');
zlabel('Z');
%pcshow fonksiyonun genel kullanýmýný öðrendik

figure();
pcshow(JAX066PC3);
title('Data Denemesi 1');
xlabel('X');
ylabel('Y');
zlabel('Z');
%Doðrudan indirilen datalar üzerinden iþlem yhapmayý baþardýk.
%deneme 1 yukarýda.

figure();
pcshow(OMA252PC3);
title('Data Denemesi 2');
xlabel('X');
ylabel('Y');
zlabel('Z');
%Dogryudan farklý x y z kordinatlarý için datalarý kullanraka iþlem yapmayý
%baþardýk.

%https://www.youtube.com/watch?v=EeruwKeOClI
%video linki yukarýdaki linktir.
%Import data ettikten sonra numeric kýsmýna alýp sadece 3 stün üzerinden
%iþlemleri yapmaya devam ettik. 3 stün x y z kordinatlarý manasýna
%gelmektedir.

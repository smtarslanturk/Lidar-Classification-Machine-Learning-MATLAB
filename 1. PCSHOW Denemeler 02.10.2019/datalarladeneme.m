A=[x(:),y(:),z(:)];
figure();
pcshow(A);
title('Sphere with Color Texture');
xlabel('X');
ylabel('Y');
zlabel('Z');
%pcshow fonksiyonun genel kullan�m�n� ��rendik

figure();
pcshow(JAX066PC3);
title('Data Denemesi 1');
xlabel('X');
ylabel('Y');
zlabel('Z');
%Do�rudan indirilen datalar �zerinden i�lem yhapmay� ba�ard�k.
%deneme 1 yukar�da.

figure();
pcshow(OMA252PC3);
title('Data Denemesi 2');
xlabel('X');
ylabel('Y');
zlabel('Z');
%Dogryudan farkl� x y z kordinatlar� i�in datalar� kullanraka i�lem yapmay�
%ba�ard�k.

%https://www.youtube.com/watch?v=EeruwKeOClI
%video linki yukar�daki linktir.
%Import data ettikten sonra numeric k�sm�na al�p sadece 3 st�n �zerinden
%i�lemleri yapmaya devam ettik. 3 st�n x y z kordinatlar� manas�na
%gelmektedir.

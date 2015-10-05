
samples = [1400 5500 10200 18400];
retrieval_time = [0.02 0.06 0.15 0.27];
plot(a,b, 'b-o'); set(gca, 'FontSize', 15);
grid on;
xlabel('# of samples');
ylabel('Retrieval time in sec.');

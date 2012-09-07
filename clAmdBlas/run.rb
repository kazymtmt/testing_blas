Dir.mkdir "dat" unless File.exist? "dat"

order = 1
stride = 1
error_check = 0

5.times do
  [0,1].each do |transa|
    #[0,1].each do |transb|
    [0].each do |transb|
      ["dgemm","sgemm"].each do |prog|
        outfile = "dat/clAmdBlas1.8.269_#{`hostname`.chop}_#{prog}_"
        #outfile = "dat/clAmdBlas1.7.257_#{`hostname`.chop}_#{prog}_"
        outfile += (order == 0) ? "C" : "R"
        outfile += (transa == 0) ? "N" : "T"
        outfile += (transb == 0) ? "N" : "T"
        outfile += ".txt"
        puts outfile
        #max_size = (prog == "dgemm") ? 7200 : 8192
        max_size = (prog == "dgemm") ? 5120 : 6144
        system "./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check} | tee -a #{outfile}"
      end
    end
  end
end


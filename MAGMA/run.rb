Dir.mkdir "dat" unless File.exist? "dat"

stride = 16
error_check = 0

5.times do
  #[0,1].each do |transa|
  [0].each do |order|
    [0,1].each do |transa|
      [0,1].each do |transb|
        ["sgemm","dgemm"].each do |prog|
          #outfile = "dat/cublas4.2.9_#{`hostname`.chop}_#{prog}_"
          #outfile = "dat/cublas5.0rc_#{`hostname`.chop}_#{prog}_"
          outfile = "dat/magma-1.2.1_m2090_#{prog}_"
          outfile += (order == 0) ? "C" : "R"
          outfile += (transa == 0) ? "N" : "T"
          outfile += (transb == 0) ? "N" : "T"
          outfile += ".txt"
          puts outfile
          open(outfile, "a") do |w|
	    max_size = (prog == "dgemm") ? 10500 : 12500
	    #system "./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check} | tee -a #{outfile}"
	    res = `./#{prog} #{order} #{transa} #{transb} #{max_size} #{stride} #{error_check}`
            w << res
          end
        end
      end
    end
  end
end


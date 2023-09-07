#ifdef _OPENACC
   integer :: mygpu, myrealgpu, num_devices
   integer(kind=acc_device_kind) :: my_device_type
   character(10) :: arg1

   my_device_type = acc_device_nvidia

   !if(argc == 1) mygpu = 0; else mygpu = atoi(argv[1]);
   if( command_argument_count() .gt. 0 )then
        call get_command_argument( 1, arg1 )
        read(arg1,'(i10)') mygpu
    else
        mygpu = 0
    endif

   call acc_set_device_type(my_device_type) 
   num_devices = acc_get_num_devices(my_device_type) 
   write(0,*) "Number of devices available: ",num_devices
   call acc_set_device_num(mygpu,my_device_type)
   write(0,*) "Trying to use GPU: ",mygpu
   myrealgpu = acc_get_device_num(my_device_type)
   write(0,*) "Actually I am using GPU: ",myrealgpu
   if(mygpu .ne. myrealgpu) then
       write(0,*) "I cannot use the requested GPU: ",mygpu
       STOP
   endif
#endif

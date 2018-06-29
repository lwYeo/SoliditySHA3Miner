#pragma once
#include <string>

namespace CudaSolver
{
	template<class T>
	public ref class ManagedObject
	{
	protected:
		T * m_Instance;

	public:
		ManagedObject(T* instance) : m_Instance(instance)
		{

		}

		virtual ~ManagedObject()
		{
			if (m_Instance != nullptr) delete m_Instance;
		}

		!ManagedObject()
		{
			if (m_Instance != nullptr) delete m_Instance;
		}

		T* GetInstance()
		{
			return m_Instance;
		}

		static std::string ToNativeString(System::String^ managedString)
		{
			using namespace System::Runtime::InteropServices;
			const char* chars = (const char*)(Marshal::StringToHGlobalAnsi(managedString)).ToPointer();
			std::string nString = chars;

			Marshal::FreeHGlobal(System::IntPtr((void*)chars));
			return nString;
		}

		static System::String ^ ToManagedString(std::string const& stdString)
		{
			return gcnew System::String(stdString.c_str());
		}
	};
}

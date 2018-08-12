#include "solver.h"

namespace CPUSolver
{
	Solver::Solver(System::String ^maxDifficulty, System::String ^threads, System::String ^kingAddress) :
		ManagedObject(new cpuSolver(ToNativeString(maxDifficulty), ToNativeString(threads), ToNativeString(kingAddress)))
	{
		m_managedOnGetSolutionTemplate = gcnew OnGetSolutionTemplateDelegate(this, &Solver::OnGetSolutionTemplate);
		System::IntPtr getSolutionTemplateStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnGetSolutionTemplate);
		cpuSolver::GetSolutionTemplateCallback getSolutionTemplateFnPtr = static_cast<cpuSolver::GetSolutionTemplateCallback>(getSolutionTemplateStubPtr.ToPointer());
		m_Instance->setGetSolutionTemplateCallback(getSolutionTemplateFnPtr);
		System::GC::KeepAlive(m_managedOnGetSolutionTemplate);

		m_managedOnMessage = gcnew OnMessageDelegate(this, &Solver::OnMessage);
		System::IntPtr messageStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnMessage);
		cpuSolver::MessageCallback messageFnPtr = static_cast<cpuSolver::MessageCallback>(messageStubPtr.ToPointer());
		m_Instance->setMessageCallback(messageFnPtr);
		System::GC::KeepAlive(m_managedOnMessage);
		
		m_managedOnSolution = gcnew OnSolutionDelegate(this, &Solver::OnSolution);
		System::IntPtr solutionStubPtr = System::Runtime::InteropServices::Marshal::GetFunctionPointerForDelegate(m_managedOnSolution);
		cpuSolver::SolutionCallback solutionFnPtr = static_cast<cpuSolver::SolutionCallback>(solutionStubPtr.ToPointer());
		m_Instance->setSolutionCallback(solutionFnPtr);
		System::GC::KeepAlive(m_managedOnSolution);
	}

	Solver::~Solver()
	{
		try { m_Instance->~cpuSolver(); }
		catch(...) {}
	}

	void Solver::OnGetSolutionTemplate(uint8_t *%solutionTemplate)
	{
		OnGetSolutionTemplateHandler(solutionTemplate);
	}

	unsigned int Solver::getLogicalProcessorsCount()
	{
		return cpuSolver::getLogicalProcessorsCount();
	}

	System::String ^Solver::getNewSolutionTemplate(System::String ^kingAddress)
	{
		return ToManagedString(cpuSolver::getNewSolutionTemplate(ToNativeString(kingAddress)));
	}

	void Solver::setCustomDifficulty(uint32_t customDifficulty)
	{
		m_Instance->setCustomDifficulty(customDifficulty);
	}

	void Solver::setSubmitStale(bool submitStale)
	{
		m_Instance->m_SubmitStale = submitStale;
	}

	bool Solver::isMining()
	{
		if (m_Instance == nullptr) return false;
		return m_Instance->isMining();
	}

	bool Solver::isPaused()
	{
		if (m_Instance == nullptr) return false;
		return m_Instance->isPaused();
	}

	void Solver::updatePrefix(System::String^ prefix)
	{
		m_Instance->updatePrefix(ToNativeString(prefix));
	}

	void Solver::updateTarget(System::String^ target)
	{
		m_Instance->updateTarget(ToNativeString(target));
	}

	void Solver::updateDifficulty(System::String^ difficulty)
	{
		m_Instance->updateDifficulty(ToNativeString(difficulty));
	}

	void Solver::startFinding()
	{
		m_Instance->startFinding();
	}

	void Solver::stopFinding()
	{
		m_Instance->stopFinding();
	}

	void Solver::pauseFinding(bool pauseFinding)
	{
		m_Instance->pauseFinding(pauseFinding);
	}

	uint64_t Solver::getTotalHashRate()
	{
		if (m_Instance == nullptr) return 0ull;
		return m_Instance->getTotalHashRate();
	}

	uint64_t Solver::getHashRateByThreadID(unsigned int const threadID)
	{
		if (m_Instance == nullptr) return 0ull;
		return m_Instance->getHashRateByThreadID(threadID);
	}

	void Solver::OnMessage(int threadID, System::String^ type, System::String^ message)
	{
		OnMessageHandler(threadID, type, message);
	}

	void Solver::OnSolution(System::String^ digest, System::String^ address, System::String^ challenge, System::String^ difficulty, System::String^ target, System::String^ solution, bool isCustomDifficulty)
	{
		OnSolutionHandler(digest, address, challenge, difficulty, target, solution, isCustomDifficulty);
	}
}

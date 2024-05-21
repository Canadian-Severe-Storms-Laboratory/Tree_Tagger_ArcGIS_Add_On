#pragma once

namespace Console {

	typedef void(__stdcall* ConsoleCallback)(const char* msg);

	ConsoleCallback consoleCallback;

	void allocConsole() {
		AllocConsole();
		freopen("CONOUT$", "w", stdout);
	}
}